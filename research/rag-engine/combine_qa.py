"""Script to combine separate QA parquet files into a single QA dataset."""

import pandas as pd
from pathlib import Path
import logging
from autorag.data.qa.schema import QA, Corpus, Raw

def combine_qa_datasets(corpus_path="data/corpus.parquet", 
                       concept_path="data/qa_concept.parquet",
                       factoid_path="data/qa_factoid.parquet", 
                       two_hop_path="data/qa_two_hop.parquet",
                       output_path="data/qa_combined.parquet"):
    """
    Combine separate QA datasets into a single dataset.
    
    Args:
        corpus_path: Path to the corpus parquet file
        concept_path: Path to the concept completion QA parquet file
        factoid_path: Path to the factoid single hop QA parquet file
        two_hop_path: Path to the two hop incremental QA parquet file
        output_path: Path where the combined QA dataset will be saved
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    
    # Load the corpus
    corpus_df = pd.read_parquet(corpus_path)
    raw_df = corpus_df[["doc_id", "contents"]].rename(columns={"doc_id": "raw_id"}).drop_duplicates(subset="raw_id").reset_index(drop=True)
    
    raw_instance = Raw(raw_df)
    corpus_instance = Corpus(corpus_df, raw_instance)
    
    # Load individual QA datasets
    qa_dfs = []
    
    for path, name in [(concept_path, "concept"), (factoid_path, "factoid"), (two_hop_path, "two_hop")]:
        path_obj = Path(path)
        if path_obj.exists():
            df = pd.read_parquet(path)
            logging.info(f"Loaded {name} QA dataset with {len(df)} rows")
            qa_dfs.append(df)
        else:
            logging.warning(f"{name} QA dataset not found at {path}")
    
    if not qa_dfs:
        raise ValueError("No QA datasets found to combine")
    
    # Combine all QA dataframes
    combined_df = pd.concat(qa_dfs, ignore_index=True)
    
    # Create QA object and save
    qa_combined = QA(combined_df, corpus_instance)
    
    # Save the combined dataset
    qa_combined.to_parquet(output_path, corpus_path)
    logging.info(f"Combined QA dataset saved with {len(qa_combined.data)} total rows to {output_path}")
    
    # Print summary
    print("\nSummary of combined QA dataset:")
    print(f"Total QA pairs: {len(qa_combined.data)}")
    if 'query_type' in combined_df.columns:
        type_counts = combined_df['query_type'].value_counts()
        print("Distribution by type:")
        for query_type, count in type_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"  {query_type}: {count} ({percentage:.1f}%)")
    
    return qa_combined

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine separate QA datasets into a single dataset.")
    parser.add_argument("--corpus-path", default="data/corpus.parquet", help="Path to corpus parquet file")
    parser.add_argument("--concept-path", default="data/qa_concept.parquet", help="Path to concept completion QA parquet file")
    parser.add_argument("--factoid-path", default="data/qa_factoid.parquet", help="Path to factoid single hop QA parquet file")
    parser.add_argument("--two-hop-path", default="data/qa_two_hop.parquet", help="Path to two hop incremental QA parquet file")
    parser.add_argument("--output-path", default="data/qa_combined.parquet", help="Path where combined QA dataset will be saved")
    
    args = parser.parse_args()
    
    combine_qa_datasets(
        corpus_path=args.corpus_path,
        concept_path=args.concept_path,
        factoid_path=args.factoid_path,
        two_hop_path=args.two_hop_path,
        output_path=args.output_path
    )