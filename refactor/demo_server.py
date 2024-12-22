import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template_string

from sentence_transformers import SentenceTransformer
from utils.logger_utils import setup_logger

app = Flask(__name__)

# Globals loaded once, then used in our Flask routes.
EMBED_MODEL = None
DOC_EMBEDDINGS = None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query", "")
        if not query.strip():
            return render_template_string(TEMPLATE, query="", results=[])

        # Embed the user’s query
        query_vector = EMBED_MODEL.encode([query])[0]

        # Calculate cosine similarity to each doc
        results = []
        for doc in DOC_EMBEDDINGS:
            doc_vec = np.array(doc["embedding"], dtype=np.float32)
            sim = cosine_similarity(query_vector, doc_vec)
            results.append((doc["id"], doc["text"], sim))

        # Sort by similarity descending & take top 5
        results.sort(key=lambda x: x[2], reverse=True)
        top_results = results[:5]

        return render_template_string(TEMPLATE, query=query, results=top_results)
    else:
        return render_template_string(TEMPLATE, query="", results=[])

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# A slightly more polished Bootstrap-based HTML template
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Document Search Demo</title>
    <meta charset="utf-8">
    <link rel="stylesheet"
      href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
  </head>
  <body class="bg-light">
    <div class="container" style="max-width: 700px;">
      <h1 class="my-4">Document Search Demo</h1>
      <form method="post" class="mb-4">
        <div class="input-group">
          <input type="text" name="query" class="form-control" placeholder="Enter search term..."
                 aria-label="Search" value="{{ query }}">
          <div class="input-group-append">
            <button class="btn btn-primary" type="submit">Search</button>
          </div>
        </div>
      </form>
      {% if results %}
        <h2 class="mt-4">Results</h2>
        <ol>
          {% for (doc_id, doc_text, sim) in results %}
          <li class="mb-3">
            <div><strong>Doc ID:</strong> {{ doc_id }}</div>
            <div><strong>Similarity:</strong> {{ "{:.4f}".format(sim) }}</div>
            <div><strong>Document:</strong> {{ doc_text }}</div>
          </li>
          {% endfor %}
        </ol>
      {% endif %}
    </div>
  </body>
</html>
"""

def load_resources(cfg, logger):
    """
    Loads the finetuned embedding model and the doc embeddings from disk,
    as specified in the Hydra config.
    """
    global EMBED_MODEL, DOC_EMBEDDINGS

    model_path = cfg.paths.finetuned_model_path
    embeddings_path = cfg.paths.doc_embeddings_path

    logger.info(f"Loading finetuned model from: {model_path}")
    EMBED_MODEL = SentenceTransformer(model_path)

    logger.info(f"Loading doc embeddings from: {embeddings_path}")
    with open(embeddings_path, "rb") as f:
        DOC_EMBEDDINGS = pickle.load(f)

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    # 1) Setup logger
    logger = setup_logger(f'{cfg.paths.output_dir}/demo_server')
    run_output_dir = Path(logger.handlers[0].baseFilename).parent
    logger.info("Starting demo server script.")

    # 2) Save the config used
    config_copy_path = run_output_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to: {config_copy_path}")

    # 3) Load resources
    load_resources(cfg, logger)

    # 4) Run the Flask server
    host = cfg.server.host  # e.g. "0.0.0.0"
    port = cfg.server.port  # e.g. 5000
    debug_mode = cfg.server.debug
    logger.info(f"Launching Flask app on {host}:{port}, debug={debug_mode}...")
    app.run(host=host, port=port, debug=debug_mode)

if __name__ == "__main__":
    main()
