from preprocessing import preprocess_file
from nltk.translate.meteor_score import meteor_score

def compute_meteor(reference_path, hypothesis_path):
    references = preprocess_file(reference_path)
    hypotheses = preprocess_file(hypothesis_path)

    scores = []

    for ref, hyp in zip(references, hypotheses):
        # meteor_score expects strings, not token lists
        ref_str = " ".join(ref)
        hyp_str = " ".join(hyp)
        score = meteor_score([ref_str], hyp_str)
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
