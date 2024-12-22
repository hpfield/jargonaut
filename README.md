# Jargonaut

**Jargonaut** is a repository illustrating how to:

* **Generate synthetic Q&A data** from domain-specific text (using a Large Language Model)
* **Finetune an embeddings model** on that data for improved semantic retrieval of jargon-heavy content
* **Build a minimal Flask web app** to demonstrate the search experience end-to-end

By leveraging **Hydra** for centralized configuration, **LLMs** for question generation, and **SentenceTransformers** for embeddings finetuning, Jargonaut offers a practical, **domain-adaptive** approach to searching specialized documents.


## Table of Contents



1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Setup](#setup)
4. [Usage](#usage)
   * [End-to-End Pipeline](#end-to-end-pipeline)
   * [Individual Scripts](#individual-scripts)
5. [Additional Notes](#additional-notes)
6. [License](#license)


## Overview

Many organizations handle **jargon-heavy** or **domain-specific** documents where off-the-shelf embeddings fall short. Jargonaut aims to **bridge that gap** by:




1. **Creating Synthetic Q&A**: We prompt an LLM to generate realistic user queries for each document.
2. **Finetuning**: We train an embeddings model on these Q&A pairs, enhancing its understanding of specialized terminology.
3. **Building Document Embeddings**: We compute embeddings for an entire corpus using our finetuned model.
4. **Serving a Local Search Demo**: A Flask-based UI allows users to enter a query and see top-matching documents (with truncated text, metadata, etc.).


## Repository Structure

Below is a simplified view of the directory layout. Some subdirectories contain additional scripts, logs, or metadata files not shown in detail:


```
jargonaut/
├── eda/
│   ├── policy-data.ipynb
│   └── token-stats.ipynb
├── environment.yml
├── govuk-policy-qa-pairs/
│   ├── data/
│   ├── policy_papers.json
│   ├── policy_papers_small.json
│   └── README.md
├── legislation_2021-2023_qa/
│   ├── test_all_3qs.csv
│   └── train_all_3qs.csv
├── llama-models/
│   ├── models/...
│   └── ...
├── README.md                <-- You are here
└── src/
    ├── build_embeddings.py  <-- Script to build & save embeddings
    ├── config/
    │   └── config.yaml      <-- Main Hydra config
    ├── demo_server.py       <-- Flask app for searching
    ├── finetune.py          <-- Finetunes embeddings
    ├── generate_qa.py       <-- Synthetic Q&A generation
    ├── outputs/             <-- Timestamped output directories
    ├── prompts/
    │   └── custom_qa_generate_prompt.txt
    ├── run_all.py           <-- Example runner (alternative to pipeline_runner)
    └── utils/
        ├── data_utils.py
        ├── llm_utils.py
        ├── logger_utils.py
        └── ...
```


### Key Directories

* `eda/`: Notebooks exploring or analyzing the data (EDA = Exploratory Data Analysis).
* `govuk-policy-qa-pairs/` & `legislation_2021-2023_qa/`: Example data directories with specialized text.
* `llama-models/`: Contains local Llama code and references (if you use the Llama-based generator).
* `src/outputs/`: Where each script’s run logs, config snapshots, and artifacts (like `train_dataset.json`, `finetuned_model/`, `doc_embeddings.pkl`) are stored in timestamped folders.

## Setup


1. **Clone this repository**:

```javascript
git clone https://github.com/YourUsername/jargonaut.git
cd jargonaut
```


2\. **Create and activate the conda environment**:

```javascript
conda env create -f environment.yml
conda activate jargonaut
```

This installs PyTorch, SentenceTransformers, Flask, Hydra, and other dependencies listed in `environment.yml`.


3\. **Adjust any paths** in `src/config/config.yaml`:

```
paths:
  data_file: "../govuk-policy-qa-pairs/policy_papers_small.json"
  output_dir: "outputs"
  prompt_file: "prompts/custom_qa_generate_prompt.txt"
  # e.g., these get updated dynamically or you can set them manually
  train_dataset_path: "outputs/generate_qa/<timestamp>/train_dataset.json"
  val_dataset_path:   "outputs/generate_qa/<timestamp>/val_dataset.json"
  finetuned_model_path: "outputs/finetune/<timestamp>/finetuned_model"
  doc_embeddings_path:   "outputs/build_embeddings/<timestamp>/doc_embeddings.pkl"
```


## Usage

### End-to-End Pipeline

`run_all.py` automates the entire process. For example, run everything in sequence:

```
cd src
python run_all.py --stage=all
```

This:


1. **Generates Q&A pairs** from your data (via LLM).
2. **Finetunes** a SentenceTransformers model on the Q&A.
3. **Builds embeddings** for your entire corpus.
4. **Launches the demo_server** to let you test queries in the browser.

**Run a single stage** if you only need that portion:

```
python run_all.py --stage=generate_qa
python run_all.py --stage=finetune
python run_all.py --stage=build_embeddings
python run_all.py --stage=demo_server
```

After each stage, `pipeline_runner.py` looks at the newly created timestamped output folder and updates your config with the relevant paths (e.g., `finetuned_model_path`).


### Individual Scripts

If you prefer a more manual approach:


1. **Generate Q&A** with `generate_qa.py`:

   ```
   cd src
   python generate_qa.py
   ```

Creates `train_dataset.json` and `val_dataset.json`.


2. **Finetune** with `finetune.py`:

   ```
   python finetune.py
   ```

   Loads Q&A data, trains a SentenceTransformers embedding model, and saves it in `finetuned_model/`.
3. **Build Embeddings** with `build_embeddings.py`:

```
python build_embeddings.py
```

Encodes your entire corpus into embeddings (saved in `doc_embeddings.pkl`).


4. **Serve** with `demo_server.py`:

```javascript
python demo_server.py
```

Starts a **Flask** server on `http://127.0.0.1:5000`. Enter a query, see the top matches, truncated text, and metadata (headers/URLs).

**Note**: Running scripts individually requires you to manually update paths in `src/config/config.yaml` (or pass Hydra overrides) so each script can find the output from the previous steps.


---

## Additional Notes

* **LLM Integration**: If you use the `Llama` generator, ensure your `ckpt_dir` and `tokenizer_path` in `config.yaml` are valid.
* **Data Exploration**: The `eda/` folder has notebooks (like `policy-data.ipynb`) for exploring the dataset.
* **Custom Data**: Replace `paths.data_file` with your own domain-specific JSON or CSV. Then retrain the pipeline for your specialized text.





