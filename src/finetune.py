import hydra
from omegaconf import DictConfig, OmegaConf
import json
from pathlib import Path
import torch
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from utils.logger_utils import setup_logger
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.schema import TextNode

def evaluate_embeddings(logger: logging.Logger, val_dataset, embed_model, top_k=5):
    from llama_index.core import VectorStoreIndex
    corpus = val_dataset.corpus
    queries = val_dataset.queries
    relevant_docs = val_dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in queries.items():
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids

        eval_results.append({
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        })
    return eval_results


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    This script handles finetuning on already-generated Q&A pairs.
    1) Reads train_dataset.json and val_dataset.json from disk
    2) Finetunes a SentenceTransformers model
    3) Optionally evaluates on val_dataset.
    """

    logger = setup_logger(f'{cfg.paths.output_dir}/finetune')
    run_output_dir = logger.handlers[0].baseFilename.rsplit("/", 1)[0]
    logger.info("Starting finetune script.")

    # Save the config used
    config_copy_path = Path(run_output_dir) / "config.yaml"
    with open(config_copy_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to: {config_copy_path}")

    train_dataset_path = cfg.paths.train_dataset_path
    val_dataset_path   = cfg.paths.val_dataset_path

    # Load the train/val datasets
    logger.info(f"Loading train dataset from: {train_dataset_path}")
    train_dataset = EmbeddingQAFinetuneDataset.from_json(train_dataset_path)
    logger.info(f"Loading val dataset from: {val_dataset_path}")
    val_dataset = EmbeddingQAFinetuneDataset.from_json(val_dataset_path)

    # Finetune
    logger.info("Starting finetuning process.")
    torch.cuda.empty_cache()

    finetune_engine = SentenceTransformersFinetuneEngine(
        dataset=train_dataset,
        model_id="BAAI/bge-small-en",
        model_output_path=str(Path(run_output_dir) / "finetuned_model"),
        val_dataset=val_dataset,
        epochs=cfg.training.epochs,
        device='cuda',  
    )
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()

    # Optional evaluate
    logger.info("Evaluating finetuned model on validation dataset.")
    val_results = evaluate_embeddings(logger, val_dataset, embed_model)
    hit_rate = sum([r["is_hit"] for r in val_results]) / len(val_results) if len(val_results) > 0 else 0
    logger.info(f"Validation set hit rate: {hit_rate}")

    logger.info("Finetune script complete.")


if __name__ == "__main__":
    main()
