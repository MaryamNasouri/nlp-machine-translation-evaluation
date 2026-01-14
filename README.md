# nlp-machine-translation-evaluation
# NLP Machine Translation Evaluation Project

## Project Overview
This project evaluates machine translation outputs using a grey-box evaluation approach.  
Two machine translation models are compared against a human reference translation using standard NLP evaluation metrics.

The goal is to demonstrate:
- Metric-based model evaluation
- Lexical and semantic comparison
- Research-style NLP analysis

---

## Problem Statement
Given:
- A human reference translation
- Two machine translation outputs (Model A and Model B)

We aim to answer:
> Which model produces higher-quality translations?

Quality is evaluated using multiple complementary metrics.

---

## Dataset
The dataset consists of:
- `references.txt` – Human reference translations  
- `hypothesis_model_a.txt` – Model A translations  
- `hypothesis_model_b.txt` – Model B translations  

The dataset is small and controlled to focus on evaluation methodology rather than model training.

---

## Preprocessing
A preprocessing pipeline is applied to normalize text:
- Lowercasing
- Punctuation removal
- Whitespace normalization
- Tokenization

Implemented in `preprocessing.py`.

---

## Evaluation Metrics

### BLEU
- Precision-based n-gram overlap metric  
- Measures lexical similarity  

### METEOR
- Recall-oriented metric  
- Accounts for synonym matches  
- More aligned with human judgment  

### ROUGE
- Overlap-based sequence metrics  
- Includes ROUGE-1, ROUGE-2, ROUGE-L  

### BERTScore
- Embedding-based semantic similarity metric  
- Uses pretrained BERT models  

---

## Results Summary

| Metric | Model A | Model B |
|-------|---------|---------|
| BLEU | Higher | Lower |
| METEOR | **0.598** | 0.341 |
| ROUGE-1 F1 | **0.643** | 0.504 |
| ROUGE-2 F1 | **0.403** | 0.165 |
| ROUGE-L F1 | **0.643** | 0.473 |
| BERTScore F1 | **0.974** | 0.947 |

### Interpretation
- Lexical metrics favor Model A due to closer phrasing to the reference
- BERTScore shows both models achieve strong semantic similarity
- Model B produces acceptable paraphrases but with lower lexical overlap

---

## Project Structure
nlp-machine-translation-evaluation/
│
├── preprocessing.py

├── evaluate_bleu.py

├── evaluate_meteor.py

├── evaluate_rouge.py

├── evaluate_bertscore.py

├── comparison_notebook.ipynb

├── data/

│ ├── references.txt

│ ├── hypothesis_model_a.txt

│ └── hypothesis_model_b.txt

├── requirements.txt

└── README.md


---

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt

Run evaluations:
python evaluate_bleu.py
python evaluate_meteor.py
python evaluate_rouge.py
python evaluate_bertscore.py

## Key Takeaways

Implemented a complete NLP evaluation pipeline

Compared MT models using lexical and semantic metrics

Demonstrated grey-box evaluation methodology

Applied research-style model comparison
