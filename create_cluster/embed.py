import random
from transformers import RobertaTokenizer, RobertaModel
from transformers import T5Tokenizer, T5EncoderModel
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import torch
import numpy as np
from functools import partial
import tqdm
from argparse import ArgumentParser
import os
import json


device = 'cuda' if torch.cuda.is_available() else 'cpu'
BSIZE = 32
SAVE_EVERY = 10000
TOTAL_NUM = 1000000


def roberta_embed(model_tokenizer, sentences):
    model, tokenizer = model_tokenizer
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model(**inputs).pooler_output
        return outputs.cpu().numpy()

    
def t5embed(model_tokenizer, sentences):
    model, tokenizer = model_tokenizer
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = torch.mean(model(**inputs).last_hidden_state, dim=1)
        return outputs.cpu().numpy()

    
def bert_embed(model_tokenizer, sentences):
    model, tokenizer = model_tokenizer
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model(**inputs).pooler_output
        return outputs.cpu().numpy()

def embed_sentences(embed_func, sentences, bsize=BSIZE, save_dir=None):
    embeddings, texts = [], []
    save_threshold = [i * SAVE_EVERY for i in range(1, TOTAL_NUM // SAVE_EVERY + 2)]
    for i in tqdm.trange(0, len(sentences), bsize):
        sentence_batch = sentences[i:i+bsize]
        embeddings.extend(embed_func(sentence_batch))
        texts.extend(sentence_batch)
        finished_count = i + bsize
        if save_dir is not None and finished_count > save_threshold[0]:
            embeddings = np.array(embeddings)
            np.save(os.path.join(save_dir, f'{finished_count}.npy'), embeddings)
            json.dump(texts, open(os.path.join(save_dir, f'{finished_count}.json'), 'w'))
            save_threshold.pop(0)
            embeddings = []
            texts = []
    if len(embeddings) > 0:
        np.save(os.path.join(save_dir, f'{finished_count}.npy'), np.concatenate(embeddings, axis=0))
        json.dump(texts, open(os.path.join(save_dir, f'{finished_count}.json'), 'w'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--subpart_name', type=str)

    args = parser.parse_args()

    model_name = args.model_name
    subpart_name = args.subpart_name

    if 'roberta' in model_name:
        model = RobertaModel.from_pretrained(model_name).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model_tokenizer = (model, tokenizer)
        embed_func = partial(roberta_embed, model_tokenizer)
    elif 't5' in model_name:
        model = T5EncoderModel.from_pretrained(model_name).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model_tokenizer = (model, tokenizer)
        embed_func = partial(t5embed, model_tokenizer)
    elif 'bert' in model_name:
        model = BertModel.from_pretrained(model_name).to(device)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model_tokenizer = (model, tokenizer)
        embed_func = partial(bert_embed, model_tokenizer)
    
    # data is a list of strings
    if subpart_name == 'wikitext-103-raw-v1':
        data = load_dataset('wikitext', subpart_name)['train']['text']
    elif subpart_name == 'wikitext-2-raw-v1':
        data = load_dataset('wikitext', subpart_name)['train']['text']

    save_dir = f'embeddings/model_name={model_name}_subpart={subpart_name}'
    os.makedirs(save_dir, exist_ok=True)
    filtered_data = [x.strip() for x in data if len(x.strip()) > 0]
    random.shuffle(filtered_data)
    filtered_data = filtered_data[:TOTAL_NUM]

    embeddings = embed_sentences(embed_func, filtered_data, save_dir=save_dir)