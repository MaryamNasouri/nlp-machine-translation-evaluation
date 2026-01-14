from preprocessing import preprocess_file
from nltk.translate.meteor_score import meteor_score

def compute_meteor(reference_path, hypothesis_path):
    references = preprocess_file(reference_path)   # list[list[str]]
    hypotheses = preprocess_file(hypothesis_path)  # list[list[str]]

    scores = []
    for ref_tokens, hyp_tokens in zip(references, hypotheses):
        # meteor_score expects tokenized inputs (Iterable[str]) in newer NLTK versions
        score = meteor_score([ref_tokens], hyp_tokens)
        scores.append(score)

    return sum(scores) / len(scores)

if __name__ == "__main__":
    ref_path = "data/references.txt"
    hyp_a_path = "data/hypothesis_model_a.txt"
    hyp_b_path = "data/hypothesis_model_b.txt"

    meteor_a = compute_meteor(ref_path, hyp_a_path)
    meteor_b = compute_meteor(ref_path, hyp_b_path)

    print(f"METEOR (Model A): {meteor_a:.4f}")
    print(f"METEOR (Model B): {meteor_b:.4f}")
