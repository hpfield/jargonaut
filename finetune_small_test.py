#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import Optional
import fire
import json
import pandas as pd
from tqdm import tqdm
# from llama_index import Document  # Updated import
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.finetuning import SentenceTransformersFinetuneEngine, generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.embeddings.openai import OpenAIEmbedding
# from models.llama3.api.datatypes import UserMessage
# from models.llama3.reference_impl.generation import Llama
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split  # Import for train/test split
import torch  # Added import

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Assuming the 'models' directory is at '/home/ubuntu/OrgSync/llama-models/models'
# You can adjust this path if 'models' is located differently
MODELS_DIR = Path("/home/ubuntu/jargonaut/llama-models/models")

# Add the 'models' directory to sys.path
sys.path.append(str(MODELS_DIR.parent))

# Absolute path to the tokenizer model
TOKENIZER_PATH = str(MODELS_DIR / "llama3" / "api" / "tokenizer.model")

# Absolute path to the checkpoint directory
DEFAULT_CKPT_DIR = "/home/rz20505/.llama/checkpoints/Llama3.1-8B-Instruct"  # Replace with your actual path

# Set environment variables required by torch.distributed
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'  # You can choose any free port

# Import necessary classes from the models package
try:
    from models.llama3.reference_impl.generation import Llama
    from models.llama3.api.datatypes import (
        UserMessage,
        SystemMessage,
        CompletionMessage,
        StopReason
    )
except ModuleNotFoundError as e:
    logger.critical(f"Error importing Llama model modules: {e}")
    sys.exit(1)

class LlamaLLM:
    def __init__(self, ckpt_dir: str, temperature: float = 0.6, top_p: float = 0.9, max_seq_len: int = 8000, max_batch_size: int = 4, model_parallel_size: Optional[int] = None):
        tokenizer_path = str(Path(__file__).parent.parent / "llama3/api/tokenizer.model")
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )
        self.temperature = temperature
        self.top_p = top_p

    def complete(self, prompt: str) -> str:
        dialog = [UserMessage(content=prompt)]
        result = self.generator.chat_completion(dialog, temperature=self.temperature, top_p=self.top_p)
        return result.generation.content

def process_data_to_nodes(data_list, verbose=False):
    if verbose:
        print(f"Processing {len(data_list)} data points into nodes.")
    # Combine 'header' and 'content' for each item
    nodes = []
    for item in data_list:
        combined_text = f"{item['header']}\n{item['content']}"
        # Create a TextNode object
        node = TextNode(text=combined_text)
        nodes.append(node)
    if verbose:
        print(f"Created {len(nodes)} TextNodes.")
    return nodes

def finetune_embeddings(train_nodes, val_nodes, checkpoint_dir, temperature=0.6, top_p=0.9):
    llama_llm = LlamaLLM(
        ckpt_dir=checkpoint_dir,
        temperature=temperature,
        top_p=top_p
    )

    # Generate datasets
    train_dataset = generate_qa_embedding_pairs(
        llm=llama_llm, nodes=train_nodes
    )
    val_dataset = generate_qa_embedding_pairs(
        llm=llama_llm, nodes=val_nodes
    )

    train_dataset.save_json("train_dataset.json")
    val_dataset.save_json("val_dataset.json")

    # Free GPU memory by deleting llama_llm and clearing cache
    del llama_llm
    torch.cuda.empty_cache()

    # Finetuning the model
    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,
        model_id="BAAI/bge-small-en",
        model_output_path="test_model",
        val_dataset=val_dataset,
        device='cuda',
    )

    finetune_engine.finetune()
    return finetune_engine.get_finetuned_model()

def evaluate_embeddings(val_dataset, embed_model, top_k=5):
    corpus = val_dataset.corpus
    queries = val_dataset.queries
    relevant_docs = val_dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(
        nodes, embed_model=embed_model, show_progress=True
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids

        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    
    return pd.DataFrame(eval_results)

def evaluate_st(dataset, model_id, name):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, name=name
    )
    model = SentenceTransformer(model_id)
    output_path = "results/"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator(model, output_path=output_path)

def main(
    json_file: str = "../../../govuk-policy-qa-pairs/policy_papers_small.json",
    checkpoint_dir: str = os.getenv('LLAMA_CKPT_DIR', str(Path.home() / ".llama/checkpoints/Meta-Llama3.1-8B-Instruct")),
    temperature: float = 0.6,
    top_p: float = 0.9
):
    # Set CUDA memory allocation config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Load the data from the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {json_file}")

    # Perform a train/test split
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Split data into {len(train_data)} training and {len(val_data)} validation items.")

    # Process the data into nodes
    train_nodes = process_data_to_nodes(train_data, verbose=True)
    val_nodes = process_data_to_nodes(val_data, verbose=True)

    # Run finetuning (using a smaller subset for demonstration)
    embed_model = finetune_embeddings(train_nodes[:5], val_nodes[:5], checkpoint_dir, temperature, top_p)

    # # Evaluate the finetuned model
    # val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")
    # val_results = evaluate_embeddings(val_dataset, embed_model)
    # hit_rate_finetuned = val_results["is_hit"].mean()

    # print(f"Finetuned model hit rate: {hit_rate_finetuned}")

    # # Optionally run Information Retrieval Evaluation with sentence-transformers
    # evaluate_st(val_dataset, "test_model", "finetuned")

if __name__ == "__main__":
    main()
