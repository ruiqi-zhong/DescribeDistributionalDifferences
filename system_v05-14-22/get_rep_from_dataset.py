import json
import os
import pickle as pkl
from typing import List

import numpy as np

import tqdm

from get_extreme_w_highlight import return_extreme_values
from datasets import load_dataset


def get_rep(
    pos: List[str],  # a list of text samples from D_1
    neg: List[str],  # a list of text samples from D_0
    pair: str = "",
    save_folder=None,
):
    if save_folder is None:
        save_folder = os.path.join("end2end_jobs", pair)
        print(save_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    else:
        print("Folder %s exists" % save_folder)

    # get samples that are representative of the differences between two distributions
    extreme_vals = return_extreme_values(pos, neg, use_shap=True)
    with open(os.path.join(save_folder, "shap_result.json"), "w") as f:
        out = json.dumps(extreme_vals, indent=4)
        f.write(out)


if __name__ == "__main__":
    imdb = load_dataset("imdb")
    imdb = imdb["train"]

    neg = [imdb[i]["text"] for i in range(imdb.num_rows) if imdb[i]["label"] == 0]
    pos = [imdb[i]["text"] for i in range(imdb.num_rows) if imdb[i]["label"] == 1]

    get_rep(pos[:100], neg[:100], pair="sentiment_analysis")
