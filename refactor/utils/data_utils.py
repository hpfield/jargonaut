import logging
from llama_index.core.schema import TextNode

def process_data_to_nodes(data_list, logger: logging.Logger = None):
    if logger:
        logger.info(f"Processing {len(data_list)} items into TextNode objects.")
    nodes = []
    for item in data_list:
        combined_text = f"{item['header']}\n{item['content']}"
        node = TextNode(text=combined_text)
        nodes.append(node)
    if logger:
        logger.info(f"Created {len(nodes)} TextNodes.")
    return nodes
