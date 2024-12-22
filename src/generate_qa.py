#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig, OmegaConf
import json
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

# Import from your utils
from utils.logger_utils import setup_logger
from utils.data_utils import process_data_to_nodes
from utils.llm_utils import (
    get_generator,
    generate_qa_embedding_pairs_chat_style,
)


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    This script handles the Q&A pair generation using your LLM.
    We:
    1) Load data and split into train/val sets.
    2) Use the LLM to generate Q&A pairs for each subset.
    3) Save out train_dataset.json and val_dataset.json.
    """

    # 1. Setup logging & output dir
    logger = setup_logger(f'{cfg.paths.output_dir}/generate_qa')
    run_output_dir = logger.handlers[0].baseFilename.rsplit("/", 1)[0]
    logger.info("Starting generate_qa script.")

    # 2. Save the exact config used
    config_copy_path = Path(run_output_dir) / "config.yaml"
    with open(config_copy_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to: {config_copy_path}")

    # 3. Load the data
    data_path = cfg.paths.data_file
    with open(data_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} items from {data_path}")

    # 4. Split data
    train_data, val_data = train_test_split(
        data,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state
    )
    logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation items.")

    # 5. Turn them into TextNodes
    train_nodes = process_data_to_nodes(train_data, logger=logger)
    val_nodes   = process_data_to_nodes(val_data,   logger=logger)

    # 6. Read custom QA prompt
    prompt_file = cfg.paths.prompt_file
    with open(prompt_file, "r") as pf:
        prompt_str = pf.read()

    system_prompt_file = cfg.paths.system_prompt_file
    with open(system_prompt_file, "r") as pf:
        system_prompt_str = pf.read()

    # 7. Instantiate the LLM generator
    logger.info("Initializing the Llama generator.")
    generator = get_generator(
        logger=logger,
        ckpt_dir=cfg.llm.ckpt_dir,
        tokenizer_path=cfg.llm.tokenizer_path,
        max_seq_len=cfg.llm.max_seq_len,
        max_batch_size=cfg.llm.max_batch_size,
        model_parallel_size=cfg.llm.model_parallel_size
    )

    # 8. Optionally reduce data for demonstration
    logger.info("Generating QA pairs for training set.")
    small_train_nodes = train_nodes[:cfg.training.train_subset_size]
    train_dataset_path = str(Path(run_output_dir) / "train_dataset.json")

    train_dataset = generate_qa_embedding_pairs_chat_style(
        logger=logger,
        nodes=small_train_nodes,
        generator=generator,
        qa_generate_prompt_str=prompt_str,
        qa_system_prompt=system_prompt_str,
        num_questions_per_chunk=cfg.training.num_questions_per_chunk,
        retry_limit=cfg.training.retry_limit,
        on_failure=cfg.training.on_failure,
        save_every=cfg.training.save_every,
        output_path=train_dataset_path,
    )

    logger.info("Generating QA pairs for validation set.")
    small_val_nodes = val_nodes[:cfg.training.val_subset_size]
    val_dataset_path = str(Path(run_output_dir) / "val_dataset.json")

    val_dataset = generate_qa_embedding_pairs_chat_style(
        logger=logger,
        nodes=small_val_nodes,
        generator=generator,
        qa_generate_prompt_str=prompt_str,
        qa_system_prompt=system_prompt_str,
        num_questions_per_chunk=cfg.training.num_questions_per_chunk,
        retry_limit=cfg.training.retry_limit,
        on_failure=cfg.training.on_failure,
        save_every=cfg.training.save_every,
        output_path=val_dataset_path,
    )

    torch.cuda.empty_cache()

    logger.info("QA pair generation complete. Exiting generate_qa script.")
    logger.info(f"Saved train dataset to: {train_dataset_path}")
    logger.info(f"Saved val dataset to:   {val_dataset_path}")


if __name__ == "__main__":
    main()
