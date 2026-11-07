"""Small CPU checks students can run before occupying a classroom GPU."""
import math
import torch
from model import GPT, GPTConfig, causal_mask
from train import prediction_windows
from evaluate import language_model_scores


def main():
    torch.set_num_threads(1)
    mask = causal_mask(4)
    assert mask.shape == (4, 4) and mask.dtype == torch.bool
    assert mask.diag().all() and mask[3, 0] and not mask[0, 3]
    sequence = torch.arange(12)
    x, y = prediction_windows(sequence, torch.tensor([0, 5]), 4)
    assert x.tolist() == [[0, 1, 2, 3], [5, 6, 7, 8]]
    assert y.tolist() == [[1, 2, 3, 4], [6, 7, 8, 9]]
    score = language_model_scores(6.0, 3)
    assert abs(score['loss'] - 2.0) < 1e-7 and abs(score['perplexity'] - math.exp(2)) < 1e-7
    model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=65, dropout=0))
    model.eval()
    a = torch.randint(65, (2, 8))
    b = a.clone()
    b[:, 4:] = (b[:, 4:] + 1) % 65
    with torch.no_grad():
        logits_a, _ = model(a, a)
        logits_b, _ = model(b, b)
    torch.testing.assert_close(logits_a[:, :4], logits_b[:, :4])
    for method in ('adamw', 'muon', 'sgdm'):
        model.configure_optimizers(0, 1e-3, (.9, .99), 'cpu', method)
    print('PASS: mask, shifted labels, scores, future-token invariance and optimizer grouping')
    print('Next: complete optimizer-step/plots TODOs and run a smoke training experiment.')


if __name__ == '__main__':
    main()
