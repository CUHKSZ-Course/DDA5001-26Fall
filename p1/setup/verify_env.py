"""Check training updates and checkpoint save/reload."""
import argparse
import io
import json
import platform

import torch
from torch import nn
from torch.nn import functional as F


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(65, 32)
        self.projection = nn.Linear(32, 32, bias=False)
        self.head = nn.Linear(32, 65)

    def forward(self, ids):
        x = self.projection(self.embedding(ids)).unsqueeze(1)
        x = F.scaled_dot_product_attention(x, x, x, is_causal=True)
        return self.head(x.squeeze(1))


def make_optimizers(model, method):
    hidden = [model.projection.weight]
    rest = [p for p in model.parameters() if p is not hidden[0]]
    return ([torch.optim.AdamW(model.parameters(), lr=1e-3)] if method == 'adamw' else [
        torch.optim.Muon(hidden, lr=1e-3, adjust_lr_fn='match_rms_adamw'),
        torch.optim.AdamW(rest, lr=1e-3),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Portable setup check; not GPU acceptance")
    args = parser.parse_args()
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: inspect driver and installed PyTorch wheel")
    device = "cpu" if args.cpu else "cuda"
    print(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": "CPU" if args.cpu else torch.cuda.get_device_name(),
    }, indent=2))
    for method in ("adamw", "muon"):
        torch.manual_seed(42)
        model = TinyModel().to(device)
        optimizers = make_optimizers(model, method)
        before = model.projection.weight.detach().clone()
        tokens = torch.randint(65, (4, 33), device=device)
        x, y = tokens[:, :-1], tokens[:, 1:]
        for _ in range(3):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=not args.cpu):
                loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            assert torch.isfinite(loss), "Nonfinite loss"
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
        assert not torch.equal(before, model.projection.weight), "No weight update"
        model.eval()
        with torch.no_grad():
            expected = model(x)
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers]}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location=device, weights_only=True)
        restored = TinyModel().to(device)
        restored.load_state_dict(checkpoint["model"])
        restored_optimizers = make_optimizers(restored, method)
        for optimizer, state in zip(restored_optimizers, checkpoint["optimizers"]):
            optimizer.load_state_dict(state)
        with torch.no_grad():
            torch.testing.assert_close(restored(x), expected)
        # Compare the next real update, with optimizer state attached to the restored model.
        for network, opts in ((model, optimizers), (restored, restored_optimizers)):
            for optimizer in opts: optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(network(x).flatten(0, 1), y.flatten())
            loss.backward()
            for optimizer in opts: optimizer.step()
        for original, recovered in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(recovered, original)
        print(f"PASS {method}: SDPA, forward/backward/update, checkpoint and resumed update agreement")
    print("CPU SETUP PASS (GPU NOT TESTED)" if args.cpu else "GPU SMOKE PASS")


if __name__ == "__main__":
    main()
