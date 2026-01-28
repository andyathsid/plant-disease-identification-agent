"""Utility script for creating AutoRAG-ready QA datasets with specific distribution."""

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
from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI

DEFAULT_CORPUS_PATH = Path("data/corpus.parquet")
DEFAULT_QA_PATH = Path("data/qa.parquet")
DEFAULT_RAW_PATH = Path("data/raw.parquet")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create QA datasets used for AutoRAG experiments with specific distribution."
    )
    parser.add_argument(
        "--qa-count",
        type=int,
        default=210,  # Default to one type's count (35% of 600)
        help="Number of QA pairs to generate for the selected type.",
    )
    parser.add_argument(
        "--query-type",
        choices=["concept", "factoid", "two_hop"],
        required=True,
        help="Type of queries to generate: concept (Concept Completion), factoid (Factoid Single Hop), or two_hop (Two Hop Incremental)",
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
        choices=["openai", "google"],
        default="openai",
        help="LLM provider to use (google for Gemini models, openai for GPT models).",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="Model name passed to the LLM class (default: gpt-4o-mini for OpenAI, gemini-2.5-flash for Google).",
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
        default=8,
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
        default=None,
        help="Path where the QA parquet will be written. If not specified, uses type-specific filename.",
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
    safe_max_tokens = min(max_tokens, 2048)  # Further reduced to prevent MAX_TOKENS errors

    if provider == "openai":
        return OpenAI(model=model_name, max_tokens=safe_max_tokens)

    if provider in ["google", "gemini"]:
        return GoogleGenAI(
            model=model_name,
            temperature=0,
            max_tokens=safe_max_tokens,
            request_timeout=60  # Add timeout to prevent hanging
        )

    raise ValueError(f"Unsupported provider: {provider}")


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)


def create_qa_dataset(
    corpus_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    qa_count: int,
    query_type: str,
    seed: int,
    lang: str,
    llm_provider: str,
    model_name: str,
    max_tokens: int,
    batch_size: int,
) -> QA:
    if corpus_df.empty:
        raise ValueError("Corpus is empty; cannot create QA dataset")
    if qa_count <= 0:
        raise ValueError("qa_count must be positive")

    logging.info(f"Generating {qa_count} {query_type} queries")

    raw_instance = Raw(raw_df)
    corpus_instance = Corpus(corpus_df, raw_df)

    # Helper to get a fresh LLM instance for each batch_apply to avoid loop issues
    def get_llm():
        return init_llm(llm_provider, model_name, max_tokens)

    if query_type == "concept":
        # Concept Completion (35%)
        concept_indices = range(0, qa_count)
        qa_data = (
            corpus_instance.sample(range_single_hop, idx_range=concept_indices)
            .map(reset_index)
            .make_retrieval_gt_contents()
            .batch_apply(
                concept_completion_query_gen,
                batch_size=max(1, batch_size // 2),  # Reduce batch size to prevent token errors
                llm=get_llm(),
                lang=lang,
            )
        )
        qa_data.data["query_type"] = "concept_completion"
        
    elif query_type == "factoid":
        # Factoid Single Hop (35%)
        factoid_indices = range(0, qa_count)
        qa_data = (
            corpus_instance.sample(range_single_hop, idx_range=factoid_indices)
            .map(reset_index)
            .make_retrieval_gt_contents()
            .batch_apply(
                factoid_query_gen,
                batch_size=max(1, batch_size // 2),  # Reduce batch size to prevent token errors
                llm=get_llm(),
                lang=lang,
            )
        )
        qa_data.data["query_type"] = "factoid_single_hop"
        
    elif query_type == "two_hop":
        # Two Hop Incremental (30%)
        two_hop_rows = []
        import uuid
        for i in range(qa_count):
            idx1 = i * 2
            idx2 = (i * 2) + 1
            if idx1 >= len(corpus_df) or idx2 >= len(corpus_df):
                logging.warning("Not enough documents for all two-hop questions. Stopping at %d", i)
                break
            doc_id1 = corpus_df.iloc[idx1]["doc_id"]
            doc_id2 = corpus_df.iloc[idx2]["doc_id"]
            two_hop_rows.append({
                "qid": str(uuid.uuid4()),
                "retrieval_gt": [[doc_id1], [doc_id2]]
            })

        if two_hop_rows:
            qa_temp = QA(pd.DataFrame(two_hop_rows), corpus_instance)
            qa_data = (
                qa_temp.map(reset_index)
                .make_retrieval_gt_contents()
                .batch_apply(
                    two_hop_incremental,
                    batch_size=max(1, batch_size // 2),  # Reduce batch size to prevent token errors
                    llm=get_llm(),
                    lang=lang,
                )
            )
            qa_data.data["query_type"] = "two_hop_incremental"
        else:
            raise ValueError("Not enough documents for two-hop questions")
    else:
        raise ValueError(f"Unknown query type: {query_type}")

    # Common generation for all
    qa = (
        qa_data.batch_apply(
            make_basic_gen_gt,
            batch_size=max(1, batch_size // 2),  # Reduce batch size to prevent token errors
            llm=get_llm(),
            lang=lang,
        )
        .batch_apply(
            make_concise_gen_gt,
            batch_size=max(1, batch_size // 2),  # Reduce batch size to prevent token errors
            llm=get_llm(),
            lang=lang,
        )
    )

    before = len(qa.data)
    qa = qa.filter(dontknow_filter_rule_based, lang=lang)
    logging.info(
        "Filtered %d low-quality pairs (%d -> %d)", before - len(qa.data), before, len(qa.data)
    )
    return qa


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    
    # Verify API key is available based on provider
    if args.llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
    elif args.llm_provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    corpus_df = ensure_corpus_dataset(args.corpus_path)
    raw_df = build_raw_dataframe(corpus_df)
    save_raw_dataset(raw_df, args.raw_path)
    
    qa = create_qa_dataset(
        corpus_df=corpus_df,
        raw_df=raw_df,
        qa_count=args.qa_count,
        query_type=args.query_type,
        seed=args.seed,
        lang=args.lang,
        llm_provider=args.llm_provider,
        model_name=args.llm_model,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    
    # Determine output path based on query type if not specified
    if args.qa_path is None:
        if args.query_type == "concept":
            output_path = Path("data/qa_concept.parquet")
        elif args.query_type == "factoid":
            output_path = Path("data/qa_factoid.parquet")
        elif args.query_type == "two_hop":
            output_path = Path("data/qa_two_hop.parquet")
        else:
            output_path = DEFAULT_QA_PATH
    else:
        output_path = args.qa_path
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qa.to_parquet(str(output_path), str(args.corpus_path))
    logging.info(
        "Persisted %s QA dataset (%d rows) to %s",
        args.query_type,
        len(qa.data),
        output_path,
    )


if __name__ == "__main__":
    main()