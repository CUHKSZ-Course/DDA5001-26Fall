"""Read saved experiments from a results folder."""
import json
from pathlib import Path


def load_runs(directory):
    directory = Path(directory)
    index = directory / 'sweep.json'
    if index.is_file():
        metadata = json.loads(index.read_text(encoding='utf-8'))
        if metadata.get('status', 'completed') != 'completed':
            raise ValueError('Experiments are not finished')
        trials = metadata['trials']
    else:
        trials = []
        for folder in sorted(directory.iterdir()):
            if not folder.is_dir() or not (folder / 'config.json').is_file():
                continue
            config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
            summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            trials.append({'run_dir': str(folder), 'optimizer': config['optimizer_name'],
                           'protocol': summary['budget'],
                           'returncode': 0 if summary['status'] == 'completed' else 1})
    if not trials or any(trial['returncode'] for trial in trials):
        raise ValueError('The folder must contain completed experiments')
    return trials
