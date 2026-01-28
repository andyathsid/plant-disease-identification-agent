"""Utility script for creating AutoRAG-ready QA datasets."""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

import nest_asyncio
nest_asyncio.apply()
import os
from dotenv import load_dotenv
load_dotenv()

from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import (
	make_basic_gen_gt,
	make_concise_gen_gt,
)
from autorag.data.qa.query.llama_gen_query import (
	factoid_query_gen,
	concept_completion_query_gen,
	two_hop_incremental,
)
from autorag.data.qa.sample import random_single_hop, range_single_hop
from autorag.data.qa.schema import Corpus, Raw, QA
from generate_corpus import generate_corpus
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI

DEFAULT_CORPUS_PATH = Path("data/corpus.parquet")
DEFAULT_QA_PATH = Path("data/qa.parquet")
DEFAULT_RAW_PATH = Path("data/raw.parquet")

google_api_key = os.getenv("GOOGLE_API_KEY")

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Create QA datasets used for AutoRAG experiments."
	)
	parser.add_argument(
		"--qa-count",
		type=int,
		default=1000,
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
		"--llm-provider",
		choices=["google", "openai"],
		default="google",
		help="LLM provider to use.",
	)
	parser.add_argument(
		"--llm-model",
		default="gemini-2.5-flash",
		help="Model name passed to the LLM class.",
	)
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=8192,
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
		"--query-type",
		choices=["concept", "factoid", "two_hop", "all"],
		default="all",
		help="Type of queries to generate in this run.",
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


def init_llm(provider: str, model_name: str, max_tokens: int) -> any:
	# Cap max_tokens to a sensible output limit for the LLM
	# Most models have an output limit between 4k and 8k
	# For gemini-2.5-flash, the output limit is typically 8192, but we'll use a safer value
	safe_max_tokens = min(max_tokens, 2048)  # Further reduced to prevent MAX_TOKENS errors
	if provider == "openai":
		return OpenAI(model=model_name, max_tokens=safe_max_tokens)

	# Add retry configuration for Google GenAI
	return GoogleGenAI(
		model=model_name,
		temperature=0,
		max_tokens=safe_max_tokens,
		request_timeout=60  # Add timeout to prevent hanging
	)


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
	return df.reset_index(drop=True)


def create_qa_dataset(
	corpus_df: pd.DataFrame,
	raw_df: pd.DataFrame,
	qa_count: int,
	seed: int,
	lang: str,
	llm_provider: str,
	model_name: str,
	max_tokens: int,
	batch_size: int,
	query_type: str = "all",
) -> QA:
	if corpus_df.empty:
		raise ValueError("Corpus is empty; cannot create QA dataset")
	if qa_count <= 0:
		raise ValueError("qa_count must be positive")

	# Calculate counts for each type
	concept_count = int(qa_count * 0.35)
	factoid_count = int(qa_count * 0.35)
	two_hop_count = qa_count - concept_count - factoid_count

	raw_instance = Raw(raw_df)
	corpus_instance = Corpus(corpus_df, raw_instance)
	qa_list = []

	# Helper to get a fresh LLM instance for each batch_apply to avoid loop issues
	def get_llm():
		return init_llm(llm_provider, model_name, max_tokens)

	# 1. Concept Completion (35%)
	if query_type in ["concept", "all"]:
		logging.info("Generating %d concept_completion queries", concept_count)
		try:
			concept_indices = range(0, concept_count)
			qa_concept = (
				corpus_instance.sample(range_single_hop, idx_range=concept_indices)
				.map(reset_index)
				.make_retrieval_gt_contents()
				.batch_apply(
					concept_completion_query_gen,
					batch_size=max(1, batch_size // 4),  # Reduce batch size to prevent token errors
					llm=get_llm(),
					lang=lang,
				)
			)
			qa_concept.data["query_type"] = "concept_completion"
			qa_list.append(qa_concept.data)
		except Exception as e:
			logging.error(f"Error generating concept completion queries: {e}")
			if len(qa_list) == 0:  # Only raise if this is the first query type
				raise

	# 2. Factoid Single Hop (35%)
	if query_type in ["factoid", "all"]:
		logging.info("Generating %d factoid_single_hop queries", factoid_count)
		try:
			factoid_indices = range(concept_count, concept_count + factoid_count)
			qa_factoid = (
				corpus_instance.sample(range_single_hop, idx_range=factoid_indices)
				.map(reset_index)
				.make_retrieval_gt_contents()
				.batch_apply(
					factoid_query_gen,
					batch_size=max(1, batch_size // 4),  # Reduce batch size to prevent token errors
					llm=get_llm(),
					lang=lang,
				)
			)
			qa_factoid.data["query_type"] = "factoid_single_hop"
			qa_list.append(qa_factoid.data)
		except Exception as e:
			logging.error(f"Error generating factoid queries: {e}")
			if len(qa_list) == 0:  # Only raise if this is the first query type
				raise

	# 3. Two Hop Incremental (30%)
	if query_type in ["two_hop", "all"]:
		logging.info("Generating %d two_hop_incremental queries", two_hop_count)
		try:
			two_hop_start_idx = concept_count + factoid_count
			two_hop_rows = []
			import uuid
			for i in range(two_hop_count):
				idx1 = two_hop_start_idx + (i * 2)
				idx2 = two_hop_start_idx + (i * 2) + 1
				if idx2 >= len(corpus_df):
					logging.warning("Not enough documents for all two-hop questions. Stopping at %d", i)
					break
				doc_id1 = corpus_df.iloc[idx1]["doc_id"]
				doc_id2 = corpus_df.iloc[idx2]["doc_id"]
				two_hop_rows.append({
					"qid": str(uuid.uuid4()),
					"retrieval_gt": [[doc_id1], [doc_id2]]
				})

			qa_two_hop = QA(pd.DataFrame(two_hop_rows), corpus_instance)
			qa_two_hop = (
				qa_two_hop.map(reset_index)
				.make_retrieval_gt_contents()
				.batch_apply(
					two_hop_incremental,
					batch_size=max(1, batch_size // 4),  # Reduce batch size to prevent token errors
					llm=get_llm(),
					lang=lang,
				)
			)
			qa_two_hop.data["query_type"] = "two_hop_incremental"
			qa_list.append(qa_two_hop.data)
		except Exception as e:
			logging.error(f"Error generating two-hop queries: {e}")
			if len(qa_list) == 0:  # Only raise if this is the first query type
				raise

	if not qa_list:
		raise ValueError(f"No queries generated for type: {query_type}")

	# Combine selected types
	combined_data = pd.concat(qa_list, ignore_index=True)
	qa = QA(combined_data, corpus_instance)

	# Common generation for all
	qa = (
		qa.batch_apply(
			make_basic_gen_gt,
			batch_size=max(1, batch_size // 4),  # Reduce batch size to prevent token errors
			llm=get_llm(),
			lang=lang,
		)
		.batch_apply(
			make_concise_gen_gt,
			batch_size=max(1, batch_size // 4),  # Reduce batch size to prevent token errors
			llm=get_llm(),
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
	qa = create_qa_dataset(
		corpus_df=corpus_df,
		raw_df=raw_df,
		qa_count=args.qa_count,
		seed=args.seed,
		lang=args.lang,
		llm_provider=args.llm_provider,
		model_name=args.llm_model,
		max_tokens=args.max_tokens,
		batch_size=args.batch_size,
		query_type=args.query_type,
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
