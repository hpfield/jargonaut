import hydra
from omegaconf import DictConfig, OmegaConf
import json
import shutil
from pathlib import Path
import torch

from utils.llm_utils import get_generator, finetune_embeddings, evaluate_embeddings, evaluate_st
from utils.logger_utils import setup_logger
from utils.data_utils import process_data_to_nodes
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from sklearn.model_selection import train_test_split


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Setup logging & output directory
    logger = setup_logger(cfg.paths.output_dir)
    run_output_dir = logger.handlers[0].baseFilename.rsplit("/", 1)[0]  # Directory where run.log is stored
    logger.info("Starting main script.")

    # Save the config used for this run
    config_copy_path = Path(run_output_dir) / "config.yaml"
    with open(config_copy_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to: {config_copy_path}")

    # Read data
    data_path = cfg.paths.data_file
    with open(data_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} items from {data_path}")

    # Train/test split
    train_data, val_data = train_test_split(
        data,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state
    )
    logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation items.")

    # Process into nodes
    train_nodes = process_data_to_nodes(train_data, logger=logger)
    val_nodes = process_data_to_nodes(val_data, logger=logger)

    # Read custom QA prompt
    prompt_file = cfg.paths.prompt_file
    with open(prompt_file, "r") as pf:
        prompt_str = pf.read()

    # Instantiate LLM
    generator = get_generator(
        logger=logger,
        ckpt_dir=cfg.llm.ckpt_dir,
        tokenizer_path=cfg.llm.tokenizer_path,
        max_seq_len=cfg.llm.max_seq_len,
        max_batch_size=cfg.llm.max_batch_size,
        model_parallel_size=cfg.llm.model_parallel_size
    )

    # Generate QA pairs and finetune
    logger.info("Starting finetuning process.")
    # For demonstration, you might train on a subset
    small_train_nodes = train_nodes[:cfg.training.train_subset_size]
    small_val_nodes   = val_nodes[:cfg.training.val_subset_size]

    # We pass the prompt string into generate_qa_embedding_pairs_chat_style calls
    # by monkeypatching `qa_generate_prompt_str` in finetune_embeddings or you can modify that function signature
    # For simplicity, let's do it directly here with a small local override:

    import functools
    from utils.llm_utils import generate_qa_embedding_pairs_chat_style

    def patched_generate_qa(*args, **kwargs):
        kwargs["qa_generate_prompt_str"] = prompt_str
        return generate_qa_embedding_pairs_chat_style(*args, **kwargs)

    # Actually finetune
    from llama_index.finetuning import SentenceTransformersFinetuneEngine, generate_qa_embedding_pairs
    import uuid

    logger.info("Generating QA pairs for training set.")
    train_dataset = patched_generate_qa(
        logger=logger,
        nodes=small_train_nodes,
        generator=generator,
        num_questions_per_chunk=cfg.training.num_questions_per_chunk,
        retry_limit=cfg.training.retry_limit,
        on_failure=cfg.training.on_failure,
        save_every=cfg.training.save_every,
        output_path=str(Path(run_output_dir) / "train_dataset.json"),
    )

    logger.info("Generating QA pairs for validation set.")
    val_dataset = patched_generate_qa(
        logger=logger,
        nodes=small_val_nodes,
        generator=generator,
        num_questions_per_chunk=cfg.training.num_questions_per_chunk,
        retry_limit=cfg.training.retry_limit,
        on_failure=cfg.training.on_failure,
        save_every=cfg.training.save_every,
        output_path=str(Path(run_output_dir) / "val_dataset.json"),
    )

    logger.info("QA pair generation complete. Beginning actual finetuning.")
    torch.cuda.empty_cache()

    finetune_engine = SentenceTransformersFinetuneEngine(
        dataset=train_dataset,
        model_id="BAAI/bge-small-en",
        model_output_path=str(Path(run_output_dir) / "finetuned_model"),
        val_dataset=val_dataset,
        device='cuda',
    )
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()

    # Optional evaluation
    logger.info("Evaluating finetuned model on validation dataset.")
    val_results = evaluate_embeddings(logger, val_dataset, embed_model)
    hit_rate = sum([r["is_hit"] for r in val_results]) / len(val_results) if len(val_results) > 0 else 0
    logger.info(f"Validation set hit rate: {hit_rate}")

    logger.info("Main script complete.")


if __name__ == "__main__":
    main()
