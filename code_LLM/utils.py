from typing import Dict, List
import csv
import re
import unidecode
import string
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# from nltk.corpus import stopwords
# nltk.download("stopwords")
# STOPWORDS = stopwords.words("english")
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "ain",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "couldn",
    "couldn't",
    "d",
    "did",
    "didn",
    "didn't",
    "do",
    "does",
    "doesn",
    "doesn't",
    "doing",
    "don",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn",
    "hadn't",
    "has",
    "hasn",
    "hasn't",
    "have",
    "haven",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "mightn",
    "mightn't",
    "more",
    "most",
    "mustn",
    "mustn't",
    "my",
    "myself",
    "needn",
    "needn't",
    "no",
    "nor",
    "not",
    "now",
    "o",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "re",
    "s",
    "same",
    "shan",
    "shan't",
    "she",
    "she's",
    "should",
    "should've",
    "shouldn",
    "shouldn't",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "that'll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "wasn",
    "wasn't",
    "we",
    "were",
    "weren",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
    "y",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}
PORTER = PorterStemmer()

SPECIAL_CHARS = {}
with open("special_chars.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for row in reader:
        SPECIAL_CHARS[int(row[0])] = row[1]


def merge_dict(d1: Dict, d2: Dict, merge_func):
    for k, v in d2.items():
        if k in d1:
            d1[k] = merge_func(d1[k], (v))
        else:
            d1[k] = v


# remove any non-ASCII char
def clean_str(s: str) -> str:
    s = unidecode.unidecode(s)
    if s.isupper():
        s = s.title()
    return s


def normalized_str(s: str) -> str:
    # data cleaning
    s = clean_str(s)
    s = "".join(filter(lambda x: x in set(string.printable), s))
    for ascii_code, replace_str in SPECIAL_CHARS.items():
        s = s.replace(chr(ascii_code), f"_{replace_str}_")
    s = "x" + s if s.startswith("_") else s
    s = s + "x" if s.endswith("_") else s
    s = "n_" + s if s[0].isnumeric() else s
    return s[0].lower() + s[1:]


def cameral_to_sentence(s: str) -> str:
    # check if it is in cameralCase
    is_cameral = s != s.lower() and s != s.upper() and "_" not in s
    # if that is a dummy item, do not change the case
    is_cameral = is_cameral and not s.startswith("dummy")

    # convert into sentence case
    if is_cameral and s != "":
        s = re.sub("([A-Z])", r" \1", s)
        return s[:1].upper() + s[1:].lower()
    return s


def snake_to_sentence(s: str) -> str:
    # check if it is in snake_case
    is_snake = "_" in s
    # if that is a dummy item, do not change the case
    is_snake = is_snake and not s.startswith("dummy")

    # convert into sentence case
    if is_snake and s != "":
        s = re.sub("_", " ", s)
    return s


def denormalized_str(s: str) -> str:
    s = s[2:] if s.startswith("n_") else s
    s = s[1:] if s.startswith("x_") else s
    s = s[:-1] if s.endswith("_x") else s
    for ascii_code, replace_str in SPECIAL_CHARS.items():
        s = s.replace(f"_{replace_str}_", chr(ascii_code))
    return snake_to_sentence(cameral_to_sentence(s))


def get_terms(rule: str) -> List[str]:
    rule_t = re.sub("[\\\\][\w]+", "", rule)  # remove variables
    rule_t = re.sub("[^0-9a-zA-Z _]+", " ", rule_t)
    terms = [i for i in rule_t.split(" ") if len(i) > 0]
    return terms


def word_similar_preprocessing(
    s: str, stemming: bool = False, stopping: bool = False
) -> str:
    word_tokens = word_tokenize(s)
    if stemming:
        word_tokens = [PORTER.stem(token) for token in word_tokens]
    if stopping:
        word_tokens = [token for token in word_tokens if token.lower() not in STOPWORDS]
    return " ".join(word_tokens)


def join_path(dirs: List[str]) -> str:
    return os.path.join(*dirs).replace("\\", "/")