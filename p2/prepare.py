"""Stage public benchmark assets and graders into assets/."""

import argparse
from pathlib import Path
import shutil
from urllib.request import urlopen

GOOGLE = "https://raw.githubusercontent.com/google-research/google-research/f4598eb0d4f8947b880cad470e157d37bd2f858b/"
HF = "https://huggingface.co/datasets/"
SOURCES = {
    "math.jsonl": HF + "HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl",
    "ifeval.jsonl": HF + "google/IFEval/resolve/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl",
    "google-LICENSE": GOOGLE + "LICENSE",
}
for name in ("evaluation_lib", "instructions", "instructions_registry", "instructions_util"):
    source = f"instruction_following_eval/{name}.py"
    SOURCES[source] = GOOGLE + source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--skip-nltk", action="store_true")
    args = parser.parse_args()
    for name, url in SOURCES.items():
        path = args.assets / name
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        with urlopen(url, timeout=90) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(path)
    if not args.skip_nltk:
        nltk_dir = args.assets / "nltk"
        marker = nltk_dir / "tokenizers/punkt_tab/english/abbrev_types.txt"
        if not marker.is_file():
            import nltk

            nltk_dir.mkdir(parents=True, exist_ok=True)
            if not nltk.download("punkt_tab", download_dir=str(nltk_dir), quiet=True):
                raise RuntimeError("Failed to download NLTK punkt_tab resource")


if __name__ == "__main__":
    main()
