# Distinguishing Fictional Voices: a Study of Authorship Verification Models for Quotation Attribution

This is the official repository for the Latech-Clfl 2024 (*EACL 2024*) paper ["Distinguishing Fictional Voices: a Study of Authorship Verification Models for Quotation Attribution"](https://aclanthology.org/2024.latechclfl-1.15). It contains all code and data to reproduce our results.

## Installation

### Data

Start by downloading the [Project Dialogism Novel Corpus](https://github.com/Priya22/project-dialogism-novel-corpus):

```bash
git clone https://github.com/Priya22/project-dialogism-novel-corpus.git
```

### Python Environment

Run the following commands to create an environment and install all the required packages:
```bash
python3 -m venv quote_av
. ./quote_av/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
```

## Running Experiments

The following will run the main experiments necessary to reproduce our results.
```bash
python main.py --experiment all --data_path project-dialogism-novel-corpus/data/ --model all --result_path results/
```

You can also run experiments using any model from Huggingface with the following. Note that the model must be similar to LUAR in the way it processes data.
```bash
python main.py --experiment all --data_path project-dialogism-novel-corpus/data/ --huggingface_model path/to_hgface_model  --result_path results/
```
