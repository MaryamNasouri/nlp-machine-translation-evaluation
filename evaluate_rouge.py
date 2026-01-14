from preprocessing import preprocess_file
from rouge_score import rouge_scorer

def compute_rouge(reference_path, hypothesis_path):
    references = preprocess_file(reference_path)
    hypotheses = preprocess_file(hypothesis_path)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1, r2, rl = [], [], []

    for ref_tokens, hyp_tokens in zip(references, hypotheses):
        ref = " ".join(ref_tokens)
        hyp = " ".join(hyp_tokens)

        scores = scorer.score(ref, hyp)

        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return {
        "rouge1_f1": sum(r1) / len(r1),
        "rouge2_f1": sum(r2) / len(r2),
        "rougeL_f1": sum(rl) / len(rl),
    }

if __name__ == "__main__":
    ref_path = "data/references.txt"
    hyp_a_path = "data/hypothesis_model_a.txt"
    hyp_b_path = "data/hypothesis_model_b.txt"

    rouge_a = compute_rouge(ref_path, hyp_a_path)
    rouge_b = compute_rouge(ref_path, hyp_b_path)

    print("ROUGE (Model A):", rouge_a)
    print("ROUGE (Model B):", rouge_b)
