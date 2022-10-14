import re
import json

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords


def generate_group(sorted):
    stops = set(stopwords.words("english"))
    candidates = []
    # set containing words in entire output sample
    word_set = set()
    for sentence in sorted:
        key = sentence
        # removes any punctuation from sentence
        key = re.sub("[^-9A-Za-z ]", "", key)
        # tokenize sentence
        words = word_tokenize(key)
        # set containing words in the current sentence
        key_words = set()
        for w in words:
            if not w in stops:
                key_words.add(w)
        # print(key_words)
        
        # loop through our tokenized words in the sentence. If a word is contained
        # in the overall set then we break, otherwise we add the word to our overall 
        # set.
        found = False
        for key_word in key_words:
            if key_word in word_set:
                # print("found word: ", key_word, "in key: ", key)
                found = True
                break
            else:
                word_set.add(key_word)
        # append to our output list, but we append at lesat 3 items at the beginning 
        # in case we have too many similar words. 
        if not found:
            candidates.append(sentence)
        elif len(candidates) < 3:
            candidates.append(sentence)
    return candidates


if __name__ == "__main__":
    ps = PorterStemmer()
    out = []
    for i in range(10):
        # read in data 
        data = json.load(open(f"shap_results/shap_result_{i}.json"))
        # sort positive and negative distributions by logits score
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
        # generate filtered out group and grab the first 5 samples
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
