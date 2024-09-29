# jargonaut

WIP - We use Llama-3.1 to generate synthetic Q&A pairs for jargon-heavy text data. Then, we use the synthetic data to finetune an embeddings model. By finetuning an embeddings model, we hope to improve the accessibility of the data to people without specialist knowledge of the subject matter.


### Running the code

Before running, the pythonpath needs to be set to the Llama3 repo. `cd` into the `llama-models` repo and run `export PYTHONPATH=$(git rev-parse --show-toplevel)` so that the Llama code can access everything.



### Working Files

Working files are in `llama-models/models/scripts/`