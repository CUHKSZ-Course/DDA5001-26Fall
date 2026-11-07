"""Prepare the bundled corpus; run only for a private writable data directory."""
from pathlib import Path
import pickle
import numpy as np

root = Path(__file__).parent
text = (root / 'input.txt').read_text(encoding='utf-8')
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
cut = int(len(text) * .9)
for name, part in (('train', text[:cut]), ('val', text[cut:])):
    np.asarray([stoi[ch] for ch in part], dtype=np.uint16).tofile(root / f'{name}.bin')
with (root / 'meta.pkl').open('wb') as stream:
    pickle.dump(dict(vocab_size=len(chars), stoi=stoi, itos=itos), stream)
print(f'{len(chars)} character tokens; train={cut}, val={len(text)-cut}')
