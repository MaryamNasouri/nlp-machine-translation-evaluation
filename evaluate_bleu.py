from preprocessing import preprocess_file
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu(reference_path, hypothesis_path):
    # references: list of list of tokens -> for corpus_bleu باید list of list of references باشد
    references_tokens = preprocess_file(reference_path)
    hypotheses_tokens = preprocess_file(hypothesis_path)

    # corpus_bleu expects: references = [[ref1_tokens], [ref2_tokens], ...]
    references = [[ref] for ref in references_tokens]
    hypotheses = hypotheses_tokens

    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)

    return bleu

if __name__ == "__main__":
    ref_path = "data/references.txt"
    hyp_a_path = "data/hypothesis_model_a.txt"
    hyp_b_path = "data/hypothesis_model_b.txt"

    bleu_a = compute_bleu(ref_path, hyp_a_path)
    bleu_b = compute_bleu(ref_path, hyp_b_path)

    print(f"BLEU (Model A): {bleu_a:.4f}")
    print(f"BLEU (Model B): {bleu_b:.4f}")
