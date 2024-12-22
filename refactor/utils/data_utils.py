import logging
from llama_index.core.schema import TextNode

def process_data_to_nodes(data_list, logger: logging.Logger = None):
    if logger:
        logger.info(f"Processing {len(data_list)} items into TextNode objects.")
    nodes = []
    for item in data_list:
        combined_text = f"{item['header']}\n{item['content']}"

        # We store metadata as a dictionary. 
        # The key structure can vary, but here's a straightforward approach:
        metadata = {
            "header": item.get("header", ""),
            "url": item.get("url", ""),
            "split_from_large_doc": item.get("split_from_large_doc", False),
        }

        # Create a TextNode with text + metadata
        node = TextNode(text=combined_text, extra_info=metadata)
        nodes.append(node)

    if logger:
        logger.info(f"Created {len(nodes)} TextNodes.")
    return nodes
