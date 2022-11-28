from collections import OrderedDict
import json
import os
import pickle as pkl
import random
from itertools import chain
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch
import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from torch import nn
from sklearn.metrics import auc, roc_curve
import scipy as sp

from contextlib import redirect_stdout

import shap

from shap_utils.utils import text as get_text
from rz_lexical_diversity import lexical_diversity

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrain_model = "roberta-base"

# Look at force plot for SVG, probaby look at only the red highlight? except make it blue

TOP_K = 100
NUM_FOLD = 4
bsize = 16
NUM_STEPS = 2000
PATIENCE = 5
max_length = 128


class RoBERTaSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrain_model)

    def forward(self, **inputs):
        model_outputs = self.model(**inputs)
        model_output_dict = vars(model_outputs)

        seq_lengths = torch.sum(
            inputs["attention_mask"], dim=-1).detach().cpu().numpy()
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

        seq_lengths = torch.sum(
            inputs["attention_mask"], dim=-1).detach().cpu().numpy()
        aggregated_repr = torch.einsum(
            "bs,bsh->bh", attention, last_hidden_state)

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


lsm = torch.nn.LogSoftmax(dim=-1)


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

    model = RoBERTaSeqAttn().to(device)
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
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
    # criterion = nn.NLLLoss()
    model.train()

    auc_score = 0
    no_improv = 0
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
        labels = torch.tensor([d["label"]
                              for d in train_data_dicts[:bsize]]).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        loss.backward()
        if step % 2 == 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return model, tokenizer


def evaluate(texts, model, tokenizer):
    model.eval()
    all_logits, all_highlights = [], []
    cur_start = 0
    while cur_start < len(texts):
        print(cur_start, len(texts))
        texts_ = texts[cur_start: cur_start + bsize]
        inputs = tokenizer(
            texts_,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        model_output_dict = model(**inputs)
        logits = lsm(
            model_output_dict["logits"].detach().cpu()).numpy().tolist()
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


def calculate_shapley_values(models, tokenizer, texts, text_to_model):
    bsize = 1

    model = models[0]

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

    sorted_dict = OrderedDict()
    explainer = shap.Explainer(predict, tokenizer)
    cur_start = 0
    while cur_start < len(texts):

        texts_ = texts[cur_start: cur_start + bsize]
        model = models[text_to_model[texts_[0]]]
        # print("sentence: ", texts_[0], "model index: ", text_to_model[texts_[0]])
        shap_values = explainer(texts_)
        shap_text = get_text(shap_values)

        for t in shap_text:
            sorted_dict[t] = shap_text[t]["span"]
        cur_start += bsize

    return sorted_dict


def get_lexical_diversity(pos_dict, neg_dict):
    pos_sorted = OrderedDict()
    neg_sorted = OrderedDict()

    for k, v in sorted(pos_dict.items(), key=lambda item: item[1], reverse=True):
        pos_sorted[k] = v
    for k, v in sorted(neg_dict.items(), key=lambda item: item[1], reverse=True):
        neg_sorted[k] = v

    As = [k for k in pos_sorted]
    Bs = [k for k in neg_sorted]

    out = {}
    for i in [0.05, 0.2, 1.0]:
        pos, neg = lexical_diversity(As, Bs, top_p=i, num_sentences=5)
        out[str(i)] = {"pos": pos, "neg": neg}

    return out


def train_and_eval(cv_dict):
    model, tokenizer = train(cv_dict)

    test_data_dict = list(
        chain(
            [{"input": x, "label": 1} for x in cv_dict["test_pos"]],
            [{"input": x, "label": 0} for x in cv_dict["test_neg"]],
        )
    )

    pos_eval_dict = evaluate(cv_dict["test_pos"], model, tokenizer)
    pos_logits, pos_highlights = (
        pos_eval_dict["logits"][:, 1],
        pos_eval_dict["highlights"],
    )
    neg_eval_dict = evaluate(cv_dict["test_neg"], model, tokenizer)
    neg_logits, neg_highlights = (
        neg_eval_dict["logits"][:, 0],
        neg_eval_dict["highlights"],
    )
    all_logits = []

    all_logits.extend(pos_eval_dict["logits"])
    all_logits.extend(neg_eval_dict["logits"])

    labels = [d["label"] for d in test_data_dict]
    fpr, tpr, thresholds = roc_curve(
        np.array(labels), np.array(all_logits)[:, 1])
    auc_roc = auc(fpr, tpr)

    return (
        {
            "test_pos_scores": pos_logits,
            "test_neg_scores": neg_logits,
            "auc_roc": auc_roc,
        },
        model,
        tokenizer,
    )


def eval_only(pos, neg, use_shap):
    model = RoBERTaSeq().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrain_model,
    )
    out = {}
    for fold_idx, cv_dict in enumerate(cv(pos, neg, NUM_FOLD)):
        if use_shap:
            pos_eval_out = evaluate(
                cv_dict["test_pos"], use_shap, model, tokenizer)
            print("breaking...")
            out = out | pos_eval_out
            break

    return out


def return_extreme_values(pos, neg):
    pos2score, neg2score = {}, {}
    pos2highlight, neg2highlight = {}, {}
    text2model = {}
    auc_roc = []
    models = []
    for fold_idx, cv_dict in enumerate(cv(pos, neg, NUM_FOLD)):
        test_scores, model, tokenizer = train_and_eval(cv_dict)
        models.append(model)
        auc_roc.append(test_scores["auc_roc"])
        for pos_text, score in zip(cv_dict["test_pos"], test_scores["test_pos_scores"]):
            pos2score[pos_text] = score
            text2model[pos_text] = fold_idx
        for neg_text, score in zip(
            cv_dict["test_neg"],
            test_scores["test_neg_scores"],
        ):
            neg2score[neg_text] = score
            text2model[neg_text] = fold_idx

    out = {}

    for i in range(3):
        out[str(i)] = {}
        pairs = get_lexical_diversity(pos2score, neg2score)

        # pair = 0.05, 0.20, or 1.0
        for pair in pairs:
            pos = pairs[pair]["pos"]
            neg = pairs[pair]["neg"]

            pos_shapley_highlights = calculate_shapley_values(
                models, tokenizer, pos, text2model
            )
            neg_shapley_highlights = calculate_shapley_values(
                models, tokenizer, neg, text2model
            )

            out[str(i)][str(pair)] = {
                "pos2highlight": pos_shapley_highlights,
                "neg2highlight": neg_shapley_highlights,
                "pos": pos,
                "neg": neg,
            }

    return {
        "pos2score": pos2score,
        "neg2score": neg2score,
        "top_p": out,
        "auc_roc": auc_roc,
    }
