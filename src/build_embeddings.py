import hydra
from omegaconf import DictConfig, OmegaConf
import json
import pickle
from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
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

    # 1) Load data
    data_path = cfg.paths.data_file
    with open(data_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} items from {data_path}")

    # 2) Process data into nodes or just skip if each item in data is a string
    _, val_data = train_test_split(
        data,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state
    )
    # Embed all data for demo
    logger.info(f"Embedding entire dataset of size: {len(data)}")
    nodes = process_data_to_nodes(data, logger=logger)

    # 3) Load the finetuned model
    model_path = cfg.paths.finetuned_model_path 
    logger.info(f"Loading finetuned SentenceTransformer model from: {model_path}")
    embed_model = SentenceTransformer(model_path)

    # 4) Embed each document
    #    Store in a list of dicts: [{"text": str, "embedding": [...]}]
    doc_embeddings = []
    texts = [node.get_content() for node in nodes]
    logger.info(f"Computing embeddings for {len(texts)} documents ...")
    embeddings = embed_model.encode(texts, batch_size=16, show_progress_bar=True)

    for idx, emb in enumerate(embeddings):
        metadata = nodes[idx].extra_info
        doc_embeddings.append({
            "id": nodes[idx].node_id,
            "text": texts[idx],
            "embedding": emb.tolist(),
            "header": metadata.get("header", ""),
            "url": metadata.get("url", ""),
            "split_from_large_doc": metadata.get("split_from_large_doc", False),
        })

    # 5) Save to disk
    embed_file = run_output_dir / "doc_embeddings.pkl"
    with open(embed_file, "wb") as f:
        pickle.dump(doc_embeddings, f)
    logger.info(f"Saved {len(doc_embeddings)} document embeddings to: {embed_file}")

    logger.info("build_embeddings script complete.")


if __name__ == "__main__":
    main()
