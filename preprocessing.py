import re

def normalize_text(text):
    """
    Normalize text by:
    - Lowercasing
    - Removing punctuation
    - Removing extra spaces
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)      # normalize spaces
    return text.strip()

def tokenize(text):
    """
    Simple whitespace tokenization
    """
    return text.split()

def preprocess_file(file_path):
    """
    Read a text file and return a list of tokenized sentences
    """
    processed_sentences = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            normalized = normalize_text(line)
            tokens = tokenize(normalized)
            processed_sentences.append(tokens)

    return processed_sentences
