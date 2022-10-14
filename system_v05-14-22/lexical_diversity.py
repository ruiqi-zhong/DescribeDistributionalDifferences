import re
import json

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords


def generate_group(sorted):
    stops = set(stopwords.words("english"))
    candidates = []
    word_set = set()
    for sentence in sorted:
        key = sentence
        key = re.sub("[^-9A-Za-z ]", "", key)
        words = word_tokenize(key)
        key_words = set()
        for w in words:
            if not w in stops:
                key_words.add(w)
        # print(key_words)
        found = False
        for key_word in key_words:
            if key_word in word_set:
                # print("found word: ", key_word, "in key: ", key)
                found = True
                break
            else:
                word_set.add(key_word)

        if not found:
            candidates.append(sentence)
        elif len(candidates) < 3:
            candidates.append(sentence)
    return candidates


if __name__ == "__main__":
    ps = PorterStemmer()
    out = []
    for i in range(10):
        data = json.load(open(f"shap_results/shap_result_{i}.json"))
        pos_sorted = {
            k: v
            for k, v in sorted(
                data["pos2score"].items(), key=lambda item: item[1], reverse=True
            )
        }

        neg_sorted = {
            k: v
            for k, v in sorted(
                data["neg2score"].items(), key=lambda item: item[1], reverse=True
            )
        }

        pos = generate_group(pos_sorted)[:5]
        neg = generate_group(neg_sorted)[:5]

        output_json = {
            "A": [sentence for sentence in pos],
            "A_Highlight": [data["pos2highlight"][sentence] for sentence in pos],
            "B": [sentence for sentence in neg],
            "B_Highlight": [data["neg2highlight"][sentence] for sentence in neg],
        }

        out.append(output_json)

    with open("shap_result_groups.json", "w") as f:
        out = json.dumps(out, indent=4)
        f.write(out)
