import json
import random
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
import torch
import numpy as np
from functools import partial
import tqdm

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
    for i in tqdm.trange(0, len(sentences), bsize):
        embeddings.append(embed_func(sentences[i:i + bsize]))
    return np.concatenate(embeddings, axis=0)


if __name__ == '__main__':
    model_name = 'roberta-base'
    model = RobertaModel.from_pretrained(model_name).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    model_tokenizer = (model, tokenizer)
    embed_fuc = partial(roberta_embed, model_tokenizer)

    subpart_name = 'wikitext-103-raw-v1'
    train_data = load_dataset('wikitext', subpart_name)['train']['text']
    filtered_data = [x.strip() for x in train_data if len(x.strip()) > 0]

    random.shuffle(filtered_data)
    filtered_data = filtered_data[:100000]

    embeddings = embed_sentences(embed_fuc, filtered_data)
    np.save(f'{model_name}_{subpart_name}_embeddings.npy', embeddings)



