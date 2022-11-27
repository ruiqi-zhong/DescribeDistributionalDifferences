import json
import random
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
from functools import partial

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BSIZE = 32


def roberta_embed(model_tokenizer, sentences):
    model, tokenizer = model_tokenizer
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model(**inputs).pooler_output
        return outputs.cpu().numpy()


def embed_sentences(embed_func, sentences, bsize=BSIZE):
    embeddings = []
    for i in range(0, len(sentences), bsize):
        embeddings.append(embed_func(sentences[i:i + bsize]))
    return np.concatenate(embeddings, axis=0)


if __name__ == '__main__':
    model_name = 'roberta-base'
    model = RobertaModel.from_pretrained(model_name).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    model_tokenizer = (model, tokenizer)


