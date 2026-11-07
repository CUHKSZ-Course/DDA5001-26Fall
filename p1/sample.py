"""Generate from an explicit endpoint/best checkpoint; tokenizer travels with it."""
import argparse
from pathlib import Path
import torch
from model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--prompt', default='\n')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    if args.tokens < 1 or args.temperature <= 0:
        parser.error('tokens and temperature must be positive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = GPT(GPTConfig(**checkpoint['model_args'])).to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    stoi, itos = checkpoint['tokenizer']['stoi'], checkpoint['tokenizer']['itos']
    if not args.prompt or any(c not in stoi for c in args.prompt):
        parser.error('Prompt must be nonempty and use characters in the course tokenizer')
    torch.manual_seed(args.seed)
    x = torch.tensor([[stoi[c] for c in args.prompt]], device=args.device)
    with torch.no_grad():
        ids = model.generate(x, args.tokens, temperature=args.temperature)[0].tolist()
    text = ''.join(itos[i] for i in ids)
    output = Path(args.output or str(Path(args.checkpoint).with_name('samples.txt')))
    output.write_text(text, encoding='utf-8')
    print(text)
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
