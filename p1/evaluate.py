"""Provided token-weighted evaluation. Do not replace with sequence-mean scoring."""
import math


def language_model_scores(total_nll, total_tokens):
    """Return loss/perplexity/bits_per_character from SUMMED NLL and token count."""
    if total_tokens <= 0:
        raise ValueError('Need at least one prediction token')
    loss = total_nll / total_tokens
    return {'loss': loss, 'perplexity': math.exp(loss), 'bits_per_character': loss / math.log(2)}
