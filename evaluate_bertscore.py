from preprocessing import preprocess_file
from bert_score import score

def compute_bertscore(reference_path, hypothesis_path):
    references = preprocess_file(reference_path)
    hypotheses = preprocess_file(hypothesis_path)

    ref_sentences = [" ".join(r) for r in references]
    hyp_sentences = [" ".join(h) for h in hypotheses]

    P, R, F1 = score(hyp_sentences, ref_sentences, lang="en", verbose=False)

    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

if __name__ == "__main__":
    ref_path = "data/references.txt"
    hyp_a_path = "data/hypothesis_model_a.txt"
    hyp_b_path = "data/hypothesis_model_b.txt"

    bert_a = compute_bertscore(ref_path, hyp_a_path)
    bert_b = compute_bertscore(ref_path, hyp_b_path)

    print("BERTScore (Model A):", bert_a)
    print("BERTScore (Model B):", bert_b)
