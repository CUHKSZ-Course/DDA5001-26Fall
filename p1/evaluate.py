"""Student score aggregation; the surrounding evaluator fixes the sampled windows."""
import math


def language_model_scores(total_nll, total_tokens):
    """Return loss/perplexity/bits_per_character from SUMMED NLL and token count."""
    if total_tokens <= 0:
        raise ValueError('Need at least one prediction token')
    # TODO scores: Compute token-mean NLL, perplexity and bits per character from the supplied sums.
    raise NotImplementedError("TODO: scores")
