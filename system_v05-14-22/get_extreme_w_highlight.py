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


import shap

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrain_model = "roberta-large"


TOP_K = 100
NUM_FOLD = 4
bsize = 16
NUM_STEPS = 2000
PATIENCE = 5
max_length = 128


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
        labels = torch.tensor([d["label"] for d in train_data_dicts[:bsize]]).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        loss.backward()
        if step % 2 == 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # early stopping criteria
        if step % 10 == 0:
            fpr, tpr, thresholds = roc_curve(
                labels.detach().cpu().numpy(), logits.detach().cpu().numpy()[:, 1]
            )
            new_auc_score = auc(fpr, tpr)
            print(new_auc_score)
            if new_auc_score <= auc_score:
                no_improv += 1
            else:
                no_improv = 0
                auc_score = new_auc_score

            if no_improv >= PATIENCE:
                break
    torch.save(model.state_dict(), "roberta-attention.pt")
    return model, tokenizer


def evaluate(texts, use_shap: bool, model, tokenizer):
    model.eval()
    if not use_shap:
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
    else:
        print("use shap")

        def predict(x):
            print("texts_: ", x)
            print("texts_ length: ", len(texts_))
            inputs = tokenizer.encode(
                x,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                is_split_into_words=True,
                padding=True,
            ).to(device)

            print(inputs)
            # print("inputs: ", inputs)
            model_output_dict = model(**inputs)
            logits = lsm(model_output_dict["logits"].detach().cpu()).numpy().tolist()
            print("outputs: ", logits)

            scores = (np.exp(logits).T / np.exp(logits).sum(-1)).T
            print(scores)
            val = sp.special.logit(scores)  # use one vs rest logit units
            print("val: ", val)
            return val

        all_logits, all_highlights = [], []
        cur_start = 0
        explainer = shap.Explainer(predict, tokenizer)
        while cur_start < len(texts):
            texts_ = texts[cur_start : cur_start + bsize]
            print("texts_ in while loop: ", texts_)
            print("texts_ length: ", len(texts_))

            shap_values = explainer(texts_)
            # shap.plots.text(shap_values)

            print("shap_values: ", shap_values)
            cur_start += bsize


def train_and_eval(cv_dict, use_shap):
    model, tokenizer = train(cv_dict)
    pos_eval_dict = evaluate(cv_dict["test_pos"], use_shap, model, tokenizer)
    pos_logits, pos_highlights = (
        pos_eval_dict["logits"][:, 1],
        pos_eval_dict["highlights"],
    )
    neg_eval_dict = evaluate(cv_dict["test_neg"], use_shap, model, tokenizer)
    neg_logits, neg_highlights = (
        neg_eval_dict["logits"][:, 0],
        neg_eval_dict["highlights"],
    )

    return {
        "test_pos_scores": pos_logits,
        "test_neg_scores": neg_logits,
        "test_pos_highlight": pos_highlights,
        "test_neg_highlight": neg_highlights,
    }


def eval_only(pos, neg, use_shap):
    print("eval_only")
    model = RoBERTaSeq().to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
    for fold_idx, cv_dict in enumerate(cv(pos, neg, NUM_FOLD)):
        pos_eval_dict = evaluate(cv_dict["test_pos"], use_shap, model, tokenizer)

    return


def return_extreme_values(pos, neg, train: bool):
    pos2score, neg2score = {}, {}
    pos2highlight, neg2highlight = {}, {}

    for fold_idx, cv_dict in enumerate(cv(pos, neg, NUM_FOLD)):
        test_scores = train_and_eval(cv_dict, False)
        for pos_text, score, highlight in zip(
            cv_dict["test_pos"],
            test_scores["test_pos_scores"],
            test_scores["test_pos_highlight"],
        ):
            pos2score[pos_text] = score
            pos2highlight[pos_text] = highlight
        for neg_text, score, highlight in zip(
            cv_dict["test_neg"],
            test_scores["test_neg_scores"],
            test_scores["test_neg_highlight"],
        ):
            neg2score[neg_text] = score
            neg2highlight[neg_text] = highlight
    return {
        "pos2score": pos2score,
        "neg2score": neg2score,
        "pos2highlight": pos2highlight,
        "neg2highlight": neg2highlight,
    }
