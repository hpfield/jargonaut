import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import tiktoken
import json
from tqdm import tqdm
from pathlib import Path

from utils.logger_utils import setup_logger

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Script for preparing a dataset by splitting large documents based on token count.
    1) Loads raw data from a JSON file (cfg.data_preparation.raw_data_file).
    2) Splits large docs into multiple parts if they exceed 'max_token_threshold'.
    3) Applies 'token_overlap' when creating chunks.
    4) Saves a small-docs-only file (cfg.data_preparation.small_file_path).
    5) Saves the combined (original+split) file (cfg.data_preparation.output_file_path).
    """

    # 1) Set up logging
    logger = setup_logger(f'{cfg.paths.output_dir}/prepare_data')
    run_output_dir = Path(logger.handlers[0].baseFilename).parent
    logger.info("Starting data preparation script.")

    # 2) Save config used
    config_copy_path = run_output_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to: {config_copy_path}")

    # 3) Grab paths from config
    raw_data_path = cfg.data_preparation.raw_data_file            # e.g. ../govuk-policy-qa-pairs/policy_papers.json
    small_file_path = cfg.data_preparation.small_file_path        # e.g. ../govuk-policy-qa-pairs/policy_papers_small.json
    output_file_path = cfg.data_preparation.output_file_path      # e.g. ../govuk-policy-qa-pairs/policy_papers_truncated.json

    # 4) Grab token settings from config (data_preparation section)
    max_token_threshold = cfg.data_preparation.max_token_threshold  # e.g. 3000
    token_overlap = cfg.data_preparation.token_overlap              # e.g. 500

    logger.info(f"Loading data from: {raw_data_path}")
    df = pd.read_json(raw_data_path)
    logger.info(f"Size of dataset: {len(df)}")

    # Count tokens for each entry
    df['combined'] = df['header'] + " " + df['content']
    enc = tiktoken.get_encoding("gpt2")
    df['token_count'] = df['combined'].apply(lambda text: len(enc.encode(text)))

    # Filter large docs
    large_docs = df[df['token_count'] > max_token_threshold]
    logger.info(f"Number of documents above threshold {max_token_threshold} tokens: {len(large_docs)}")

    def split_document_iteratively(text, token_limit=4000, token_overlap=500):
        """
        Iteratively splits a document into smaller chunks based on token_limit,
        overlapping by token_overlap. Attempts to split on newline tokens.
        """
        tokens = enc.encode(text)
        chunks = []
        start_index = 0

        while start_index < len(tokens):
            end_index = min(start_index + token_limit, len(tokens))

            # Try to split at a newline if possible
            while end_index > start_index and tokens[end_index - 1] != enc.encode('\n')[0]:
                end_index -= 1

            if end_index == start_index:
                # If no newline found, force split at token_limit
                end_index = min(start_index + token_limit, len(tokens))

            chunk_tokens = tokens[start_index:end_index]
            chunk_text = enc.decode(chunk_tokens)

            # Indicate continuation
            if start_index > 0:
                chunk_text = "[Note: Continuation from previous part of the document.]\n\n" + chunk_text

            if end_index < len(tokens):
                chunk_text += "\n\n[Note: Document truncated here due to token length; see next part for continuation.]"

            chunks.append(chunk_text)

            # Move start index forward with overlap
            next_start_index = end_index - token_overlap
            if next_start_index <= start_index:
                start_index = end_index
            else:
                start_index = next_start_index

        return chunks

    chunk_stats = []
    new_entries = []

    # Process large documents
    logger.info("Splitting large documents...")
    for _, row in tqdm(large_docs.iterrows(), total=len(large_docs), desc="Processing large docs"):
        split_texts = split_document_iteratively(row['combined'], max_token_threshold, token_overlap)
        chunk_stats.append((row['url'], len(split_texts)))

        for i, part_text in enumerate(split_texts):
            header = row['header'] if i == 0 else row['header'] + " (continued)"
            new_entries.append({
                'url': row['url'],
                'header': header,
                'content': part_text,
                'split_from_large_doc': True
            })

    # Remove original large docs
    df_small_docs = df[df['token_count'] <= max_token_threshold]

    # Save the small docs to file (e.g., policy_papers_small.json)
    logger.info(f"Saving docs <= {max_token_threshold} tokens to: {small_file_path}")
    df_small_docs.to_json(small_file_path, orient='records', indent=2)

    # Merge splitted chunks with small docs
    logger.info("Combining small and splitted documents...")
    new_entries_df = pd.DataFrame(new_entries)
    df_combined = pd.concat([df_small_docs, new_entries_df], ignore_index=True)

    # Mark unsplit docs
    df_combined['split_from_large_doc'] = df_combined['split_from_large_doc'].fillna(False)

    # Recalc token counts for new dataset
    df_combined['combined'] = df_combined['header'] + " " + df_combined['content']
    df_combined['token_count'] = df_combined['combined'].apply(lambda text: len(enc.encode(text)))

    # Save final dataset
    logger.info(f"Saving combined (small + splitted) dataset to: {output_file_path}")
    df_combined[['url', 'header', 'content', 'split_from_large_doc']].to_json(
        output_file_path, orient='records', indent=2
    )

    # Display chunk stats
    chunk_stats_df = pd.DataFrame(chunk_stats, columns=['url', 'num_chunks'])
    logger.info(f"Chunk stats:\n{chunk_stats_df.describe()}")

    # Final token stats
    most_tokens = df_combined['token_count'].max()
    least_tokens = df_combined['token_count'].min()
    average_tokens = df_combined['token_count'].mean()
    logger.info(f"Most tokens after split: {most_tokens}")
    logger.info(f"Least tokens after split: {least_tokens}")
    logger.info(f"Average tokens after split: {average_tokens:.2f}")

    logger.info("Data preparation script complete.")

if __name__ == "__main__":
    main()
