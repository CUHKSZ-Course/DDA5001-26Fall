"""Optional batch runner. Defaults to the optimizer and LR in the chosen config."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from configurator import load_config
from train import DEFAULTS
from checkpoint_io import run_monitored, KernelHealthError


def resolve_config_path(filename):
    path = Path(filename)
    if not path.is_file():
        path = Path(__file__).parent / path
    return path.resolve(strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train_shakespeare_char.py')
    parser.add_argument('--optimizers', default=None)
    parser.add_argument('--multipliers', default=None)
    parser.add_argument('--protocol', choices=['tokens', 'time', 'both'], default='tokens')
    parser.add_argument('--tokens', type=int, default=None, help='Override max_tokens; otherwise use the selected config')
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--devices', default='0', help='Physical GPU IDs, comma separated; or cpu for smoke checks')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--output', required=True, help='New sweep directory')
    args = parser.parse_args()
    args.config = str(resolve_config_path(args.config))
    config = load_config(DEFAULTS, [args.config])
    if args.tokens is None:
        args.tokens = config['max_tokens']
    if args.optimizers is None:
        args.optimizers = config['optimizer_name']
    if args.multipliers is None:
        args.multipliers = str(config['lr_multiplier'])
    methods, multipliers = args.optimizers.split(','), [float(v) for v in args.multipliers.split(',')]
    devices = args.devices.split(',')
    if args.jobs < 1 or any(not math.isfinite(v) or v <= 0 for v in multipliers) or any(m not in ('adamw', 'muon', 'sgdm') for m in methods):
        parser.error('Invalid jobs, optimizer or LR multipliers')
    if (args.protocol != 'time' and args.tokens <= 0) or (args.protocol != 'tokens' and (not math.isfinite(args.seconds) or args.seconds <= 0)):
        parser.error('Selected protocol requires a positive budget')
    if args.protocol != 'tokens' and args.jobs > len(set(devices)):
        parser.error('Fixed-time comparisons require separate GPUs for concurrent runs; use jobs=1 on one GPU')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocols = ['tokens', 'time'] if args.protocol == 'both' else [args.protocol]
    trials = list(itertools.product(protocols, methods, multipliers))
    lanes = [[] for _ in range(args.jobs)]
    for index, trial in enumerate(trials):
        lanes[index % args.jobs].append((index, trial))
    results = []
    record_lock = threading.Lock()
    def save_sweep(status):
        temporary = output / 'sweep.json.tmp'
        temporary.write_text(json.dumps({'arguments': vars(args), 'status': status,
            'planned_trials':len(trials), 'trials':results}, indent=2), encoding='utf-8')
        temporary.replace(output / 'sweep.json')
    save_sweep('running')
    def worker(lane):
        outcomes = []
        device = devices[lane % len(devices)]
        for index, (protocol, method, multiplier) in lanes[lane]:
            run_dir = output / f'{index:02d}-{method}-{protocol}-lr{multiplier:g}'
            command = [sys.executable, 'train.py', args.config, f'--optimizer_name={method}',
                f'--lr_multiplier={multiplier}', f'--out_dir={run_dir}',
                '--device=cpu' if device == 'cpu' else '--device=cuda',
                f'--max_tokens={args.tokens if protocol == "tokens" else 0}',
                f'--max_seconds={args.seconds if protocol == "time" else 0.0}']
            env = dict(os.environ)
            if device != 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = device
            failure=None
            with (output / f'{index:02d}.log').open('w', encoding='utf-8') as stream:
                try:
                    result = run_monitored(command,cwd=Path(__file__).parent,env=env,stdout=stream,stderr=subprocess.STDOUT)
                except KernelHealthError as exc:
                    failure=str(exc);print(failure,file=sys.stderr,flush=True)
                    result=subprocess.CompletedProcess(command,125)
            outcomes.append({'run_dir': str(run_dir), 'returncode': result.returncode, 'device': device,
                             'protocol': protocol, 'optimizer': method, 'lr_multiplier': multiplier,
                             **({'host_error':failure} if failure else {})})
            with record_lock:
                results.append(outcomes[-1])
                save_sweep('running')
            print(outcomes[-1], flush=True)
            if failure:break
        return outcomes
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        list(executor.map(worker, range(args.jobs)))
    save_sweep('failed' if any(row['returncode'] for row in results) else 'completed')
    raise SystemExit(1 if any(row['returncode'] for row in results) else 0)


if __name__ == '__main__':
    main()
