#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig, OmegaConf
import json
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# If you used SentenceTransformers
from sentence_transformers import SentenceTransformer

# If you're using your own data utils
from utils.data_utils import process_data_to_nodes
from utils.logger_utils import setup_logger


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger = setup_logger(f'{cfg.paths.output_dir}/build_embeddings')
    run_output_dir = Path(logger.handlers[0].baseFilename).parent
    logger.info("Starting build_embeddings script.")

    # Save the exact config used
    config_copy_path = run_output_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to: {config_copy_path}")

    # 1) Load your data
    data_path = cfg.paths.data_file
    with open(data_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} items from {data_path}")

    # 2) Process data into nodes (if you use llama_index or a custom format)
    #    Or just skip if each item in data is a string
    from sklearn.model_selection import train_test_split
    _, val_data = train_test_split(
        data,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state
    )
    # For demonstration, let's embed the entire *val_data*, or the entire data itself
    # depending on your preference
    logger.info(f"Embedding entire dataset of size: {len(data)}")
    nodes = process_data_to_nodes(data, logger=logger)

    # 3) Load the finetuned model
    model_path = cfg.paths.finetuned_model_path  # e.g. "outputs/finetune/<TIMESTAMP>/finetuned_model"
    logger.info(f"Loading finetuned SentenceTransformer model from: {model_path}")
    embed_model = SentenceTransformer(model_path)

    # 4) Embed each document
    #    We'll store them in a list of dicts: [{"text": str, "embedding": [...]}]
    doc_embeddings = []
    texts = [node.get_content() for node in nodes]  # or node.text
    logger.info(f"Computing embeddings for {len(texts)} documents ...")
    embeddings = embed_model.encode(texts, batch_size=16, show_progress_bar=True)

    for idx, emb in enumerate(embeddings):
        doc_embeddings.append({
            "id": nodes[idx].node_id,  # or idx
            "text": texts[idx],
            "embedding": emb.tolist()  # store as list if using JSON/pickle
        })

    # 5) Save to disk
    embed_file = run_output_dir / "doc_embeddings.pkl"
    with open(embed_file, "wb") as f:
        pickle.dump(doc_embeddings, f)
    logger.info(f"Saved {len(doc_embeddings)} document embeddings to: {embed_file}")

    logger.info("build_embeddings script complete.")


if __name__ == "__main__":
    main()
