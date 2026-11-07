"""Bundle completed code and lightweight experiment evidence, excluding checkpoints/data."""
import argparse
import csv
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def export_results(sweep_dir, output):
    sweep_dir, output = Path(sweep_dir).resolve(), Path(output).resolve()
    metadata = json.loads((sweep_dir/'sweep.json').read_text(encoding='utf-8'))
    trials = metadata['trials']
    if metadata.get('status', 'completed') != 'completed' or not trials or any(t['returncode'] for t in trials):
        raise ValueError('Only a completed sweep can be exported for submission')
    for trial in trials:
        run_dir = sweep_dir / Path(trial['run_dir'].replace('\\', '/')).name
        if run_dir.resolve().parent != sweep_dir:
            raise ValueError('A run directory points outside this sweep')
        config = json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
        summary = json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))
        json.loads((run_dir/'environment.json').read_text(encoding='utf-8'))
        with (run_dir/'metrics.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if summary['status'] != 'completed' or not rows:
            raise ValueError(f'Incomplete run: {run_dir.name}')
        tokens_per_update = config['batch_size'] * config['block_size'] * config['gradient_accumulation_steps']
        if (int(rows[-1]['tokens']) != summary['tokens'] or int(rows[-1]['step']) != summary['step']
                or summary['tokens'] != summary['step'] * tokens_per_update):
            raise ValueError(f'Inconsistent token/update records: {run_dir.name}')
        if config['max_tokens'] and summary['tokens'] != config['max_tokens']:
            raise ValueError(f'Token budget was not completed: {run_dir.name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Use a new archive filename')
    files = []
    # Include core and additional Python analysis scripts, but no environments/agent state.
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('data', 'runs', '__pycache__', 'node_modules')
                   and not (Path(directory)/d).is_symlink()]
        for name in names:
            p = Path(directory)/name
            if not p.is_symlink() and (p.suffix == '.py' or p == ROOT/'requirements.txt'):
                files.append((p, 'code/' + p.relative_to(ROOT).as_posix()))
    for p in sweep_dir.rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix in ('.json', '.csv', '.log', '.txt', '.png', '.pdf', '.svg'):
            if not p.resolve().is_relative_to(sweep_dir):
                raise ValueError('Evidence file points outside sweep')
            files.append((p, 'results/' + p.relative_to(sweep_dir).as_posix()))
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr('README.txt', 'Completed code and experiment evidence. Add your report PDF to the final submission.\n'
                         'Checkpoints and the supplied dataset are excluded; keep checkpoints locally for possible review.\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
    return {'archive':str(output), 'trials':len(trials), 'files':len(files), 'bytes':output.stat().st_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export_results(args.sweep_dir, args.output), indent=2))
