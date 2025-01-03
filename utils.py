from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModel,
)
import tqdm
import torch
import torch.nn.functional as F
import numpy as np


def stat_with_nones(l, stat="mean"):
    if stat == "mean":
        return np.mean([i for i in l if i is not None])
    elif stat == "std":
        return np.std([i for i in l if i is not None])


def luar_tokenize(tokenizer, quotes, batch_first=False, max_length=64):
    tokens = tokenizer(
        quotes,
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    )

    if not batch_first:
        tokens["input_ids"] = tokens["input_ids"].reshape(1, -1, max_length)
        tokens["attention_mask"] = tokens["attention_mask"].reshape(1, -1, max_length)
    else:
        tokens["input_ids"] = tokens["input_ids"].reshape(-1, 1, max_length)
        tokens["attention_mask"] = tokens["attention_mask"].reshape(-1, 1, max_length)

    return tokens


def get_model(model_name, is_hgface=False):
    if is_hgface : 
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )
        return model, tokenizer
    elif model_name == "semantics":
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        model.max_seq_length = 64
        return model, None
    elif model_name == "stel":
        model = SentenceTransformer("AnnaWegmann/Style-Embedding")
        model.max_seq_length = 64
        return model, None
    elif model_name == "emotions":
        tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        config = AutoConfig.from_pretrained("SamLowe/roberta-base-go_emotions")
        model = AutoModelForSequenceClassification.from_pretrained(
            "SamLowe/roberta-base-go_emotions", config=config
        )
        return model, tokenizer
    
    elif model_name == "luar":
        tokenizer = AutoTokenizer.from_pretrained(
            "rrivera1849/LUAR-MUD", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "rrivera1849/LUAR-MUD", trust_remote_code=True
        )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     "rrivera1849/LUAR-CRUD", trust_remote_code=True
        # )
        # model = AutoModel.from_pretrained(
        #     "rrivera1849/LUAR-CRUD", trust_remote_code=True
        # )
        return model, tokenizer
    
    elif model_name == "uar_scene":
        tokenizer = AutoTokenizer.from_pretrained(
            "gasmichel/UAR_scene", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "gasmichel/UAR_scene", trust_remote_code=True
        )
        return model, tokenizer
    elif model_name == "uar_play" : 
        tokenizer = AutoTokenizer.from_pretrained(
            "gasmichel/UAR_Play", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "gasmichel/UAR_Play", trust_remote_code=True
        )
    else:
        raise ValueError("Model must be one of semantics, stel, emotions, luar, uar_scene or uar_play")

def process_quotes(quotes, model_name, model, tokenizer):
    if model_name in ["stel", "semantics"]:
        quote_embeddings = []

        for quotes in tqdm.tqdm(quotes):
            quote_embeddings.append(
                model.encode(
                    quotes,
                    device=model.device,
                    normalize_embeddings=True,
                    convert_to_numpy=False,
                    convert_to_tensor=True,
                )
            )
    else:
        quote_embeddings = []

        batch_size = 32
        with torch.no_grad():
            for idx, novel_quotes in enumerate(tqdm.tqdm(quotes)):
                last_h = []

                model.eval()
                for idx in range(0, len(novel_quotes), batch_size):
                    tokens = tokenizer(
                        novel_quotes[idx : idx + batch_size],
                        return_tensors="pt",
                        truncation=True,
                        max_length=64,
                        padding=True,
                    )
                    u = model(
                        **tokens.to(model.device),
                        return_dict=True,
                        output_hidden_states=True
                    )
                    last_h.append(u.hidden_states[-1][:, 0].detach().cpu())
                quote_embeddings.append(F.normalize(torch.cat(last_h), dim=-1))
    return quote_embeddings
