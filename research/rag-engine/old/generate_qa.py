"""Utility script for creating AutoRAG-ready QA datasets."""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import (
	make_basic_gen_gt,
	make_concise_gen_gt,
)
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop
from autorag.data.qa.schema import Corpus, Raw, QA
from generate_corpus import generate_corpus
from llama_index.llms.openai import OpenAI

DEFAULT_CORPUS_PATH = Path("data/corpus.parquet")
DEFAULT_QA_PATH = Path("data/qa.parquet")
DEFAULT_RAW_PATH = Path("data/raw.parquet")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Create QA datasets used for AutoRAG experiments."
	)
	parser.add_argument(
		"--qa-count",
		type=int,
		default=200,
		help="Number of QA pairs to sample before filtering.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed used during sampling.",
	)
	parser.add_argument(
		"--lang",
		choices=["en", "ko", "ja"],
		default="en",
		help="Language prompts used during QA generation.",
	)
	parser.add_argument(
		"--llm-model",
		default="gpt-4o-mini",
		help="Model name passed to llama_index.llms.openai.OpenAI.",
	)
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=512,
		help="Maximum tokens for each LLM completion.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=16,
		help="Concurrency used by AutoRAG batch_apply calls.",
	)
	parser.add_argument(
		"--corpus-path",
		type=Path,
		default=DEFAULT_CORPUS_PATH,
		help="Path to corpus parquet file.",
	)
	parser.add_argument(
		"--qa-path",
		type=Path,
		default=DEFAULT_QA_PATH,
		help="Path where the QA parquet will be written.",
	)
	parser.add_argument(
		"--raw-path",
		type=Path,
		default=DEFAULT_RAW_PATH,
		help="Path where the helper raw parquet will be written.",
	)
	return parser.parse_args()


def ensure_corpus_dataset(corpus_path: Path) -> pd.DataFrame:
	if corpus_path.exists():
		return pd.read_parquet(corpus_path)
	if corpus_path != DEFAULT_CORPUS_PATH:
		raise FileNotFoundError(f"Missing corpus dataset at {corpus_path}")
	logging.info("Corpus parquet missing, generating from JSONL sources...")
	return generate_corpus()


def build_raw_dataframe(corpus_df: pd.DataFrame) -> pd.DataFrame:
	return (
		corpus_df[["doc_id", "contents"]]
		.rename(columns={"doc_id": "raw_id"})
		.drop_duplicates(subset="raw_id")
		.reset_index(drop=True)
	)


def save_raw_dataset(raw_df: pd.DataFrame, raw_path: Path) -> None:
	raw_path.parent.mkdir(parents=True, exist_ok=True)
	raw_df.to_parquet(raw_path)
	logging.info("Saved raw dataset to %s", raw_path)


def init_llm(model_name: str, max_tokens: int) -> OpenAI:
	return OpenAI(model=model_name, temperature=0.0, max_tokens=max_tokens)


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
	return df.reset_index(drop=True)


def create_qa_dataset(
	corpus_df: pd.DataFrame,
	raw_df: pd.DataFrame,
	qa_count: int,
	seed: int,
	lang: str,
	llm: OpenAI,
	batch_size: int,
) -> QA:
	if corpus_df.empty:
		raise ValueError("Corpus is empty; cannot create QA dataset")
	if qa_count <= 0:
		raise ValueError("qa_count must be positive")
	sample_size = min(qa_count, len(corpus_df))
	if sample_size < qa_count:
		logging.warning(
			"Requested %s QA pairs but only %s documents available; adjusting sample",
			qa_count,
			sample_size,
		)
	raw_instance = Raw(raw_df)
	corpus_instance = Corpus(corpus_df, raw_instance)
	qa = (
		corpus_instance.sample(
			random_single_hop,
			n=sample_size,
			random_state=seed,
		)
		.map(reset_index)
		.make_retrieval_gt_contents()
		.batch_apply(
			factoid_query_gen,
			batch_size=batch_size,
			llm=llm,
			lang=lang,
		)
		.batch_apply(
			make_basic_gen_gt,
			batch_size=batch_size,
			llm=llm,
			lang=lang,
		)
		.batch_apply(
			make_concise_gen_gt,
			batch_size=batch_size,
			llm=llm,
			lang=lang,
		)
	)
	before = len(qa.data)
	qa = qa.filter(dontknow_filter_rule_based, lang=lang)
	logging.info(
		"Filtered %s low-quality pairs (%s -> %s)", before - len(qa.data), before, len(qa.data)
	)
	return qa


def main() -> None:
	args = parse_args()
	logging.basicConfig(
		level=logging.INFO,
		format="%(levelname)s | %(message)s",
	)
	corpus_df = ensure_corpus_dataset(args.corpus_path)
	raw_df = build_raw_dataframe(corpus_df)
	save_raw_dataset(raw_df, args.raw_path)
	llm = init_llm(args.llm_model, args.max_tokens)
	qa = create_qa_dataset(
		corpus_df=corpus_df,
		raw_df=raw_df,
		qa_count=args.qa_count,
		seed=args.seed,
		lang=args.lang,
		llm=llm,
		batch_size=args.batch_size,
	)
	args.qa_path.parent.mkdir(parents=True, exist_ok=True)
	qa.to_parquet(str(args.qa_path), str(args.corpus_path))
	logging.info(
		"Persisted QA dataset (%s rows) to %s",
		len(qa.data),
		args.qa_path,
	)


if __name__ == "__main__":
	main()
