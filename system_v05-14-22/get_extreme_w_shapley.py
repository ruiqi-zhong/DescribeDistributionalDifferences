import sys
sys.path.append("./")
from collections import OrderedDict, defaultdict

import json
import os
import random
from itertools import chain
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, RobertaTokenizer
import torch
import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score
import argparse
import shap
from typing import List, Dict
import time
from shap_utils.utils import text as get_html_text
from gadgets.lexical_diversity import lexical_diversity
import pickle as pkl


# set up the directory
shap_result_dir = 'shap_results/'
tmp_model_dir = os.path.join(shap_result_dir, 'tmp_model')
results_dir = os.path.join(shap_result_dir, 'results')
for dir in [shap_result_dir, tmp_model_dir, results_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)


# Look at force plot for SVG, probaby look at only the red highlight? except make it blue
NUM_FOLD = 4
bsize = 16
NUM_STEPS = 2000
max_length = 128
RESAMPLE = 5
REPEAT_FOR_SHAP = 2
DEBUG = False

# hyperparameters for debugging
if DEBUG:
    NUM_STEPS = 300
    NUM_FOLD = 2
    RESAMPLE = 1
    REPEAT_FOR_SHAP = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrain_model = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(pretrain_model)

lsm = torch.nn.LogSoftmax(dim=-1)
class RoBERTaSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrain_model)

    def forward(self, **inputs):
        model_outputs = self.model(**inputs)
        model_output_dict = vars(model_outputs)

        seq_lengths = torch.sum(inputs["attention_mask"], dim=-1).detach().cpu().numpy()
        model_output_dict["highlight"] = [
            [1.0 / seq_length for _ in range(seq_length)] for seq_length in seq_lengths
        ]
        return model_output_dict


class RoBERTaSeqAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrain_model)
        self.clf_layer = nn.Linear(self.model.config.hidden_size, 2)
        self.attn_layer = nn.Linear(self.model.config.hidden_size, 1)
        self.sm = nn.Softmax(dim=-1)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.loss_func = nn.NLLLoss()

    def forward(self, **inputs):
        last_hidden_state = self.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).last_hidden_state

        attn_logits = self.attn_layer(last_hidden_state).squeeze(axis=-1)
        attn_logits[inputs["attention_mask"] == 0] = float("-inf")
        attention = self.sm(attn_logits)

        seq_lengths = torch.sum(inputs["attention_mask"], dim=-1).detach().cpu().numpy()
        aggregated_repr = torch.einsum("bs,bsh->bh", attention, last_hidden_state)

        logits = self.lsm(self.clf_layer(aggregated_repr))

        return_dict = {
            "logits": logits,
            "highlight": [
                attention[i][:s].detach().cpu().numpy()
                for i, s in enumerate(seq_lengths)
            ],
        }
        if "labels" in inputs:
            loss = self.loss_func(logits, inputs["labels"])
            return_dict["loss"] = loss
        return return_dict

clf_class = RoBERTaSeq


def cv(pos, neg, K):
    return [
        {
            "train_pos": [p for i, p in enumerate(pos) if i % K != k],
            "train_neg": [n for i, n in enumerate(neg) if i % K != k],
            "test_pos": [p for i, p in enumerate(pos) if i % K == k],
            "test_neg": [n for i, n in enumerate(neg) if i % K == k],
        }
        for k in range(K)
    ]


def get_spans(tokenizer, text):
    be = tokenizer(text)
    length = len(be["input_ids"])
    results = []
    for i in range(length):
        if i in (0, length - 1):
            results.append((0, 0))
        else:
            start, end = be.token_to_chars(i)
            results.append((start, end))
    return results


def train(cv_dict):
    train_data_dicts = list(
        chain(
            [{"input": x, "label": 1} for x in cv_dict["train_pos"]],
            [{"input": x, "label": 0} for x in cv_dict["train_neg"]],
        )
    )

    model = clf_class().to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 400, NUM_STEPS)
    model.train()

    for step in tqdm.trange(NUM_STEPS):
        random.shuffle(train_data_dicts)
        input_texts = [d["input"] for d in train_data_dicts[:bsize]]
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        labels = torch.tensor([d["label"] for d in train_data_dicts[:bsize]]).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]

        loss.backward()
        if step % 2 == 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return model


def evaluate(texts, model):
    model.eval()
    all_logits, all_highlights = [], []
    cur_start = 0
    while cur_start < len(texts):
        texts_ = texts[cur_start : cur_start + bsize]
        inputs = tokenizer(
            texts_,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        model_output_dict = model(**inputs)
        logits = lsm(model_output_dict["logits"].detach().cpu()).numpy().tolist()
        all_highlights.extend(model_output_dict["highlight"])
        all_logits.extend(logits)
        cur_start += bsize
    assert len(all_logits) == len(texts)
    all_spans = [get_spans(tokenizer, text) for text in texts]
    assert len(all_spans) == len(all_highlights)
    for a, b in zip(all_spans, all_highlights):
        assert len(a) == len(b) or len(a) >= max_length

    highlights = [
        {s: h for s, h in zip(spans, highlights) if s != (0, 0)}
        for spans, highlights in zip(all_spans, all_highlights)
    ]

    return {"logits": np.array(all_logits), "highlights": highlights}


def calculate_shapley_values(model, texts):

    def predict(x):
        # TODO: need to set indices based off of positive or negative results
        # print("x.toList(): ", x.tolist())
        inputs = tokenizer(
            x.tolist(),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        model_output_dict = model(**inputs)
        logits = lsm(model_output_dict["logits"].detach().cpu()).numpy()
        logits = logits[:, 1]
        return logits
    
    explainer = shap.Explainer(predict, tokenizer)
    text2shap_results = {}
    for text in tqdm.tqdm(texts):
        shap_values = explainer([text])
        text2shap_results[text] = shap_values
    return text2shap_results


def get_text_group_pairs(pos_dict: Dict[str, float], neg_dict: Dict[str, float], top_ps: List[float]):
    pos_sorted = OrderedDict()
    neg_sorted = OrderedDict()

    for k, v in sorted(pos_dict.items(), key=lambda item: item[1], reverse=True):
        pos_sorted[k] = v
    for k, v in sorted(neg_dict.items(), key=lambda item: item[1], reverse=True):
        neg_sorted[k] = v

    As = [k for k in pos_sorted]
    Bs = [k for k in neg_sorted]

    text_group_pairs = []
    for p in top_ps:
        for _ in range(RESAMPLE):
            pos, neg = lexical_diversity(As, Bs, top_p=p, num_sentences=5)
            text_group_pairs.append({"pos": pos, "neg": neg, "top_p": p})
    return text_group_pairs


def train_and_eval(cv_dict):
    model = train(cv_dict)
    pos_eval_dict = evaluate(cv_dict["test_pos"], model)
    neg_eval_dict = evaluate(cv_dict["test_neg"], model)

    all_logits = np.concatenate((pos_eval_dict["logits"], neg_eval_dict["logits"]), axis=0)[:,1]
    all_labels = np.concatenate((np.ones(len(pos_eval_dict["logits"])), np.zeros(len(neg_eval_dict["logits"]))), axis=0)

    auc = roc_auc_score(all_labels, all_logits)

    return {
            "test_pos_scores": pos_eval_dict["logits"][:, 1],
            "test_neg_scores": neg_eval_dict["logits"][:, 0],
            "auc_roc": auc,
            "test_pos_highlights": pos_eval_dict["highlights"],
            "test_neg_highlights": neg_eval_dict["highlights"],
            'model': model
        }


def get_median_render(shap_results):
    if len(shap_results) == 0:
        return None
    shap_result = shap_results[0]
    shap_result.values = np.median([x.values for x in shap_results], axis=0)
    html_text = list(get_html_text(shap_result).values())[0]['span']
    return html_text


def return_extreme_values(pos, neg, use_shap: bool, top_ps: List[float], repeat_for_shap: int = REPEAT_FOR_SHAP):
    pos2score, neg2score = {}, {}
    text2model_path = {}
    clf_scores = {}

    for fold_idx, cv_dict in enumerate(cv(pos, neg, NUM_FOLD)):
        train_and_eval_result = train_and_eval(cv_dict)
        model = train_and_eval_result['model']
        model_tmp_path = os.path.join(tmp_model_dir, f"model_{fold_idx}_{int(time.time())}.pt")
        if use_shap:
            torch.save(model.state_dict(), model_tmp_path)

        for pos_text, score in zip(cv_dict["test_pos"], train_and_eval_result["test_pos_scores"]):
            pos2score[pos_text] = score
            text2model_path[pos_text] = model_tmp_path
        for neg_text, score in zip(cv_dict["test_neg"], train_and_eval_result["test_neg_scores"]):
            neg2score[neg_text] = score
            text2model_path[neg_text] = model_tmp_path
        clf_scores[model_tmp_path] = train_and_eval_result["auc_roc"]
        print(f"fold {fold_idx} done, auc: {train_and_eval_result['auc_roc']}")

    pairs = get_text_group_pairs(pos2score, neg2score, top_ps)
    text2shap_results = defaultdict(dict)

    if use_shap:
        all_text_for_shap = set()
        for pair in pairs:
            all_text_for_shap |= set(pair["pos"])
            all_text_for_shap |= set(pair["neg"])
        model_path2texts = defaultdict(list)
        for text in all_text_for_shap:
            model_path2texts[text2model_path[text]].append(text)
        
        for model_path, texts in model_path2texts.items():
            model = clf_class()
            model.load_state_dict(torch.load(model_path))
            model.to(device)

            shap_dict = calculate_shapley_values(model, texts)
            for text, shap_values in shap_dict.items():
                text2shap_results[text][model_path] = shap_values
            if os.path.exists(model_path):
                os.remove(model_path)

    new_pos, new_neg = [[t for t in l if t not in text2shap_results] for l in [pos, neg]]
    new_cv_dicts = cv(new_pos, new_neg, NUM_FOLD * 2)
    for new_fold_idx in range(repeat_for_shap):
        new_train_and_eval_result = train_and_eval(new_cv_dicts[new_fold_idx])
        model = new_train_and_eval_result['model']
        clf_scores['for_shapley_fold%d' % new_fold_idx] = new_train_and_eval_result["auc_roc"]

        shap_dict = calculate_shapley_values(model, list(set(text2shap_results)))
        for text, shap_values in shap_dict.items():
            text2shap_results[text]['for_shapley_fold%d' % new_fold_idx] = shap_values
        print(f"new fold {new_fold_idx} done, auc: {new_train_and_eval_result['auc_roc']}")
    

    return {
        'text2shap_results': text2shap_results,
        'clf_scores': clf_scores,
        'pairs': pairs,
        'text2median_render': {text: get_median_render(list(shap_results.values())) for text, shap_results in text2shap_results.items()}
    }


def get_rep(
    pos: List[str],  # a list of text samples from D_1
    neg: List[str],  # a list of text samples from D_0
    job_name: str = "",
    save_folder=None,
    top_ps=None
):
    if top_ps is None:
        top_ps = [0.05, 0.2, 1.0]

    if save_folder is None:
        save_folder = os.path.join(results_dir, job_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    else:
        print("Folder %s exists" % save_folder)

    extreme_vals = return_extreme_values(pos, neg, use_shap=True, top_ps=top_ps)

    print("Saving to pkl...")
    pkl.dump(extreme_vals, open(os.path.join(save_folder, "shap_result.pkl"), "wb"))
    return extreme_vals



if __name__ == '__main__':
    # idx = 0
    # distr_pair = json.load(open('benchmark_sec_4/benchmark.json', 'r'))[idx]
    # pos, neg = distr_pair['positive_samples'], distr_pair['negative_samples']

    # # all_shap_results = get_rep(pos, neg, job_name='icml_benchmark_idx%d_debug' % idx)
    # # this is for debugging, obvious that the model can get auc roc 1.0 and highlitghts steve
    # neg = [x + ' steve ' for x in pos]
    # all_shap_results = get_rep(pos, neg, job_name='icml_benchmark_debug')

    parser = argparse.ArgumentParser("Trains and evaluates describe difference model")

    parser.add_argument("--percent", type=str, default="0.05,0.2,1.0")
    parser.add_argument("--data_name", type=str, default="icml_benchmark")

    args = parser.parse_args()
    top_ps = [float(f) for f in args.percent.split(",")]
    data_name = args.data_name
    pair = json.load(open('distr_pairs/%s.json' % data_name, 'r'))

    print('top fractions', top_ps)
    print('data name', data_name)

    job_name = data_name + "-time" + str(float(time.time()))


    get_rep(
        pos=pair["positive_samples"],
        neg=pair["negative_samples"],
        job_name=job_name,
        top_ps=top_ps,
    )