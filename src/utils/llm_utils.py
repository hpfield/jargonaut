import os
import sys
import logging
from pathlib import Path
import torch
import re
import uuid
import warnings
from typing import List
import tiktoken

# Ensure repo root is discoverable
repo_root = os.popen("git rev-parse --show-toplevel").read().strip()
if repo_root and repo_root not in sys.path:
    sys.path.append(repo_root)

# Set environment variables needed by PyTorch and torch.distributed
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# Add the 'models' directory to sys.path
project_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
models_dir = project_root / "llama-models/models"
if str(models_dir.parent) not in sys.path:
    sys.path.append(str(models_dir.parent))

from models.llama3.reference_impl.generation import Llama
from models.llama3.api.datatypes import (
    UserMessage,
    SystemMessage,
    CompletionMessage,
    StopReason
)

from llama_index.core.schema import MetadataMode, TextNode
from llama_index.finetuning import SentenceTransformersFinetuneEngine, generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from tqdm import tqdm


def get_generator(logger: logging.Logger, ckpt_dir: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int, model_parallel_size):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )
    return generator


def generate_qa_embedding_pairs_chat_style(
    logger: logging.Logger,
    nodes: List[TextNode],
    generator,
    qa_generate_prompt_str: str,
    qa_system_prompt: str,
    num_questions_per_chunk: int,
    retry_limit: int,
    on_failure: str,
    save_every: int,
    output_path: str,
):
    queries = {}
    corpus = {}
    relevant_docs = {}
    enc = tiktoken.get_encoding("gpt2")

    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    system_message = SystemMessage(content=qa_system_prompt)

    save_counter = 0

    for node_id, text in tqdm(node_dict.items()):
        user_prompt = qa_generate_prompt_str.format(
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
                logger.warning(f"Error querying LLM: {e}. Full prompt is {len(enc.encode(qa_system_prompt + text))} tokens. Retrying {retry_count}/{retry_limit}...")

        if not success:
            if on_failure == "fail":
                logger.error(f"Failed to query LLM after {retry_limit} retries. Aborting.")
                raise RuntimeError(f"Failed to query LLM after {retry_limit} retries.")
            elif on_failure == "continue":
                logger.warning(f"Skipping node {node_id} after {retry_limit} retries.")
                continue

        result_lines = response.split("\n")
        questions = [re.sub(r"^\d+[\).\s]", "", q).strip() for q in result_lines]
        questions = [q for q in questions if len(q) > 0][:num_questions_per_chunk]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

        corpus[node_id] = text

        save_counter += 1
        if save_counter % save_every == 0:
            dataset = EmbeddingQAFinetuneDataset(queries=queries, corpus=corpus, relevant_docs=relevant_docs)
            dataset.save_json(output_path)
            logger.info(f"Saved progress at {save_counter} entries.")

        del response
        torch.cuda.empty_cache()

    # Final save
    dataset = EmbeddingQAFinetuneDataset(queries=queries, corpus=corpus, relevant_docs=relevant_docs)
    dataset.save_json(output_path)
    logger.info("Final dataset saved.")

    return dataset
