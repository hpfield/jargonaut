#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
import uuid
import warnings
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.finetuning import SentenceTransformersFinetuneEngine, generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.embeddings.openai import OpenAIEmbedding
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer

# Import from utils
from stages.utils import get_generator, UserMessage, SystemMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_data_to_nodes(data_list, verbose=False):
    if verbose:
        print(f"Processing {len(data_list)} data points into nodes.")
    nodes = []
    for item in data_list:
        combined_text = f"{item['header']}\n{item['content']}"
        node = TextNode(text=combined_text)
        nodes.append(node)
    if verbose:
        print(f"Created {len(nodes)} TextNodes.")
    return nodes

# from stages.utils import UserMessage, SystemMessage

CUSTOM_QA_GENERATE_PROMPT_TMPL = """\
Write {num_questions_per_chunk} search queries someone might write that would be answered by this document. \
This is to create a Q&A dataset to finetune an embeddings model so that documents like these can be found more efficiently. \
You must only respond with the search queries on separate lines, nothing else. \
The document you are using is: \
{context_str}
"""

def generate_qa_embedding_pairs_chat_style(
    nodes: List['TextNode'],
    generator,  # This should be an instance of Llama from models.llama3.reference_impl.generation
    qa_generate_prompt_tmpl: str = CUSTOM_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk: int = 2,
    retry_limit: int = 3,
    on_failure: str = "continue",  # options are "fail" or "continue"
    save_every: int = 500,
    output_path: str = "qa_finetune_dataset.json",
    verbose: bool = True,
) -> 'EmbeddingQAFinetuneDataset':
    """
    Generate QA pairs from a set of nodes using a chat-based approach with a Llama generator.
    This function:
    - Uses a system and user message to prompt the model.
    - Leverages the CUSTOM_QA_GENERATE_PROMPT_TMPL.
    - Periodically saves the dataset.
    
    Args:
        nodes (List[TextNode]): List of TextNode objects to process.
        generator: A Llama instance (from models.llama3.reference_impl.generation).
        qa_generate_prompt_tmpl (str): The template for generating QA prompts.
        num_questions_per_chunk (int): Number of questions to generate per chunk.
        retry_limit (int): Number of times to retry on failure.
        on_failure (str): Action on repeated failures ('fail' or 'continue').
        save_every (int): Save dataset after processing this many nodes.
        output_path (str): File path to save the JSON output.
        verbose (bool): If True, print debugging messages.

    Returns:
        EmbeddingQAFinetuneDataset: The generated dataset.
    """
    # Load existing partial data if available
    queries = {}
    corpus = {}
    relevant_docs = {}

    # Convert nodes to a dict for easy indexing
    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    start_index = len(corpus)
    save_counter = start_index

    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Define a system message for the model
    system_message = SystemMessage(content="You are an AI assistant that generates only a specified number of search queries. You receive a document and a number of queries to generate. You must respond with only the requested number of search queries, one per line, and nothing else.")
    

    for node_id, text in tqdm(list(node_dict.items())[start_index:], initial=start_index):
        # Construct the user prompt from the template
        user_prompt = qa_generate_prompt_tmpl.format(
            context_str=text,
            num_questions_per_chunk=num_questions_per_chunk
        )

        user_msg = UserMessage(content=user_prompt)

        chat_history = [system_message, user_msg]

        retry_count = 0
        success = False
        response = None
        while retry_count < retry_limit:
            try:
                # Use the Llama generator for chat completion
                # Adjust the arguments as needed based on your generator's API
                # If your Llama generator expects a different method name or parameters, update accordingly.
                result = generator.chat_completion(
                    chat_history,
                    max_gen_len=None,
                    temperature=0.0,
                    top_p=0.9,
                )
                out_message = result.generation
                response = out_message.content.strip()
                success = True
                break
            except Exception as e:
                retry_count += 1
                if verbose:
                    print(f"Error querying LLM: {e}. Retrying {retry_count}/{retry_limit}...")

        if not success:
            # Handle failure based on user preference
            if on_failure == "fail":
                raise RuntimeError(f"Failed to query LLM after {retry_limit} retries.")
            elif on_failure == "continue":
                if verbose:
                    print(f"Skipping node {node_id} after {retry_limit} retries.")
                continue

        # Process the response to extract questions line by line
        result_lines = response.split("\n")
        questions = [re.sub(r"^\d+[\).\s]", "", q).strip() for q in result_lines]
        questions = [q for q in questions if len(q) > 0][:num_questions_per_chunk]

        num_questions_generated = len(questions)
        if num_questions_generated < num_questions_per_chunk:
            warnings.warn(
                f"Fewer questions generated ({num_questions_generated}) "
                f"than requested ({num_questions_per_chunk})."
            )

        # Store the queries and references
        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

        corpus[node_id] = text

        save_counter += 1
        if save_counter % save_every == 0:
            dataset = EmbeddingQAFinetuneDataset(
                queries=queries, corpus=corpus, relevant_docs=relevant_docs
            )
            dataset.save_json(output_path)
            if verbose:
                print(f"Saved progress at {save_counter} entries.")

        # Attempt to free GPU memory if applicable
        del response
        torch.cuda.empty_cache()

    # Save the final dataset
    dataset = EmbeddingQAFinetuneDataset(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs
    )
    dataset.save_json(output_path)
    if verbose:
        print("Final dataset saved.")

    return dataset



def finetune_embeddings(train_nodes, val_nodes, checkpoint_dir, temperature=0.6, top_p=0.9):

    generator = get_generator()

    print("Generating QA pairs for training set.")
    train_dataset = generate_qa_embedding_pairs_chat_style(
        nodes=train_nodes, generator=generator
    )
    print("Generating QA pairs for validation set.")
    val_dataset = generate_qa_embedding_pairs_chat_style(
        nodes=val_nodes, generator=generator
    )

    train_dataset.save_json("train_dataset.json")
    val_dataset.save_json("val_dataset.json")

    torch.cuda.empty_cache()

    # Finetune the model
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Finetune and evaluate embeddings with an LLM.')
    parser.add_argument('--json_file', type=str, default='../govuk-policy-qa-pairs/policy_papers_small.json',
                        help='Path to the JSON file containing data.')
    parser.add_argument('--checkpoint_dir', type=str, default=str(Path.home() / ".llama/checkpoints/Meta-Llama3.1-8B-Instruct"),
                        help='Path to the LLaMA checkpoint directory.')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for LLM generation.')
    parser.add_argument('--top_p', type=float, default=0.9, help='top_p for LLM generation.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    json_file = args.json_file
    checkpoint_dir = args.checkpoint_dir
    temperature = args.temperature
    top_p = args.top_p

    # Load the data from the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} items from {json_file}")

    # Perform a train/test split
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    logging.info(f"Split data into {len(train_data)} training and {len(val_data)} validation items.")

    # Process the data into nodes
    train_nodes = process_data_to_nodes(train_data, verbose=True)
    val_nodes = process_data_to_nodes(val_data, verbose=True)

    # Run finetuning (using a smaller subset for demonstration)
    embed_model = finetune_embeddings(train_nodes[:5], val_nodes[:5], checkpoint_dir, temperature, top_p)

    # Optional evaluation steps:
    # val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")
    # val_results = evaluate_embeddings(val_dataset, embed_model)
    # hit_rate_finetuned = val_results["is_hit"].mean()
    # logging.info(f"Finetuned model hit rate: {hit_rate_finetuned}")
    # evaluate_st(val_dataset, "test_model", "finetuned")

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
