"""Save machine evidence and derive a CUDA wheel plan; never read credentials."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import urllib.request

# Conservative Linux driver floors from NVIDIA's full toolkit compatibility rows.
# These avoid relying on limited minor-version compatibility / PTX exceptions.
PROFILES = {'cu130': (580, 65, 6), 'cu128': (570, 26, 0)}


def version_tuple(value):
    if not re.fullmatch(r'\d+(?:\.\d+){1,2}', value.strip()):
        raise ValueError(f'Unrecognized NVIDIA driver version: {value!r}')
    parts = tuple(map(int, value.strip().split('.')))
    return (parts + (0, 0, 0))[:3]


def gpu_rows(text):
    rows = []
    for row in csv.reader(text.splitlines()):
        if len(row) != 3:
            raise ValueError('Unexpected nvidia-smi CSV columns')
        name, driver, memory = [v.strip() for v in row]
        version_tuple(driver)
        rows.append({'name': name, 'driver': driver, 'memory_mib': int(memory)})
    return rows


def choose_torch(gpus, requested='auto', torch_version='2.9.1'):
    if not re.fullmatch(r'\d+\.\d+\.\d+', torch_version):
        raise ValueError('TORCH_VERSION must be an explicit numeric version')
    if requested not in ('auto', 'cpu', *PROFILES):
        raise ValueError('TORCH_FLAVOR must be auto, cu130, cu128 or cpu')
    if requested == 'cpu':
        flavor, reason = 'cpu', 'CPU explicitly selected; this does not accept a classroom GPU.'
    else:
        if not gpus:
            raise ValueError('No working NVIDIA GPU detected. Install/fix the driver and reboot, then inspect again.')
        driver = min(version_tuple(g['driver']) for g in gpus)
        if requested == 'auto':
            flavor = next((f for f, floor in PROFILES.items() if driver >= floor), None)
            if flavor is None:
                raise ValueError('Driver below this workstation profile (570.26); update driver rather than silently installing CPU torch.')
        else:
            flavor = requested
            if driver < PROFILES[flavor]:
                raise ValueError(f'{flavor} requires driver >= {PROFILES[flavor]} under this conservative policy')
        reason = f'Detected minimum driver {driver}; selected {flavor}. GPU architecture/kernels still require the real smoke test.'
    return {'torch_version': torch_version, 'flavor': flavor,
            'index': f'https://download.pytorch.org/whl/{flavor}', 'reason': reason}


def probe(argv):
    executable = shutil.which(argv[0])
    if executable is None:
        return {'command': argv, 'returncode': None, 'stdout': '', 'stderr': 'command not found'}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, errors='replace', timeout=45,
                                env={**os.environ, 'LC_ALL': 'C'})
        return {'command': argv, 'executable': executable, 'returncode': result.returncode,
                'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {'command': argv, 'returncode': None, 'stdout': '', 'stderr': 'timed out'}
    except OSError as exc:
        return {'command': argv, 'returncode': None, 'stdout': '', 'stderr': str(exc)}


def collect(data_root, requested, torch_version, stage, check_network=False, include_tex=True):
    commands = {
        'fstab': ['cat', '/etc/fstab'],
        'grub_defaults': ['cat', '/etc/default/grub'],
        'efi_boot_order': ['efibootmgr', '-v'],
        'boot_config_hashes': ['sha256sum', '/etc/fstab', '/etc/default/grub', '/boot/grub/grub.cfg'],
        'local_users': ['getent', 'passwd'],
        'local_groups': ['getent', 'group'],
        'memory_modules': ['dmidecode', '--type', '17'],
        'memory_error_correction': ['dmidecode', '--type', '16'],
        'kernel_taint': ['cat', '/proc/sys/kernel/tainted'],
        'vscode_package': ['dpkg-query', '-W', 'code'],
        'clock': ['timedatectl', 'show', '--property=Timezone', '--property=NTPSynchronized', '--property=LocalRTC'],
        'image_manager': ['ls', '-l', '/usr/share/vdts/enterUpdate', '/usr/share/vdts/leaveUpdate'],
        'pending_dpkg': ['dpkg', '--audit'],
        'suspend_policy': ['systemctl', 'is-enabled', 'sleep.target', 'suspend.target', 'hibernate.target'],
        'block_devices': ['lsblk', '--json', '-b', '-o', 'NAME,SIZE,FSTYPE,MOUNTPOINTS'],
        'nvidia_smi': ['nvidia-smi'],
        'gpu_query': ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'],
        'toolkit': ['nvcc', '--version'],
    }
    for tool in ('node', 'npm', 'codex', 'opencode', 'git', 'lab-uv', 'pdflatex', 'g++'):
        commands[tool] = [tool, '--version']
    commands['pdftoppm'] = ['pdftoppm', '-v']
    report = {'schema': 1, 'time_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
              'hostname': platform.node(), 'kernel': platform.release(), 'architecture': platform.machine(),
              'os_release': Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else platform.platform(),
              'data_root': str(Path(data_root).resolve()), 'commands': {}, 'filesystems': {}}
    for name, argv in commands.items():
        report['commands'][name] = probe(argv)
    for path in ('/', '/usr', '/var', '/opt/dda5001', data_root):
        existing = Path(path)
        while not existing.exists():
            existing = existing.parent
        mount = probe(['findmnt', '--json', '--target', str(existing), '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS'])
        capacity = shutil.disk_usage(existing)
        report['filesystems'][path] = {'observed_at': str(existing), 'mount': mount,
                                      'total_bytes': capacity.total, 'free_bytes': capacity.free}
    if Path('/proc/meminfo').exists():
        report['memory'] = '\n'.join(line for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapTotal:', 'SwapFree:')))
    try:
        query = report['commands']['gpu_query']
        report['gpus'] = gpu_rows(query['stdout']) if query['returncode'] == 0 else []
        report['torch_plan'] = choose_torch(report['gpus'], requested, torch_version)
        report['plan_error'] = None
    except ValueError as exc:
        report['torch_plan'] = None
        report['plan_error'] = str(exc)
    runtime = Path('/opt/dda5001/venv/bin/python')
    if runtime.exists():
        report['installed_torch'] = probe([str(runtime), '-c',
            'import json,torch; print(json.dumps({"torch":torch.__version__,"cuda_runtime":torch.version.cuda}))'])
    report['install_blockers'] = []
    if check_network:
        node_version=os.environ.get('NODE_VERSION','v24.22.2')
        vscode_version=os.environ.get('VSCODE_VERSION','1.145.1')
        urls = ['https://pypi.org/simple/uv/', f'https://nodejs.org/dist/{node_version}/SHASUMS256.txt',
                'https://registry.npmjs.org/opencode-ai',
                f'https://update.code.visualstudio.com/{vscode_version}/linux-deb-x64/stable']
        if report['torch_plan']:
            urls.append(report['torch_plan']['index'] + '/torch/')
        if include_tex:
            tex_mirror = os.environ.get('MIKTEX_MIRROR', 'https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/win32/miktex').rstrip('/')
            urls += [tex_mirror + '/setup/miktex.org.key',
                     tex_mirror + '/tm/packages/miktex-zzdb1-2.9.tar.lzma']
        report['downloads'] = {}
        for url in urls:
            print(f'Checking download endpoint: {url}', flush=True)
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    response.read(256)
                    report['downloads'][url] = {'status': response.status}
            except Exception as exc:
                report['downloads'][url] = {'error': str(exc)}
                report['install_blockers'].append(f'Download endpoint unavailable: {url}: {exc}')
    return report


def save_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    target = directory / f'{stamp}.json'
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    target.write_text(encoded, encoding='utf-8')
    target.chmod(0o600)
    readable = target.with_suffix('.txt')
    lines = [f"Time: {report['time_utc']}", f"Stage: {report['stage']}",
             f"Host: {report['hostname']}", f"Data root: {report['data_root']}",
             f"PyTorch plan: {json.dumps(report['torch_plan'])}", f"Plan error: {report['plan_error']}"]
    for name, result in report['commands'].items():
        lines += [f'\n[{name}] exit={result["returncode"]}', result['stdout'], result['stderr']]
    lines += ['\n[filesystems]', json.dumps(report['filesystems'], ensure_ascii=False, indent=2)]
    if 'downloads' in report:
        lines += ['\n[downloads]', json.dumps(report['downloads'], indent=2)]
    lines += ['\n[installation blockers]', '\n'.join(report.get('install_blockers', []))]
    readable.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    readable.chmod(0o600)
    temp = directory / 'latest.tmp'
    temp.write_text(encoded, encoding='utf-8')
    temp.chmod(0o600)
    temp.replace(directory / 'latest.json')
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/home')
    parser.add_argument('--output-dir', default='/var/lib/dda5001/inspection')
    parser.add_argument('--stage', default='inspect')
    parser.add_argument('--flavor', default=os.environ.get('TORCH_FLAVOR', 'auto'))
    parser.add_argument('--torch-version', default=os.environ.get('TORCH_VERSION', '2.9.1'))
    parser.add_argument('--read-plan', help='Read a saved report and print version/flavor/index, one per line')
    parser.add_argument('--network', action='store_true')
    parser.add_argument('--no-tex', action='store_true')
    args = parser.parse_args()
    if args.read_plan:
        report = json.loads(Path(args.read_plan).read_text(encoding='utf-8'))
        if report['plan_error']:
            parser.exit(1, report['plan_error'] + '\n')
        if report.get('install_blockers'):
            parser.exit(1, '\n'.join(report['install_blockers']) + '\nNo installation started; fix connectivity and rerun.\n')
        plan = report['torch_plan']
        print('\n'.join(plan[k] for k in ('torch_version', 'flavor', 'index')))
        return
    report = collect(args.data_root, args.flavor, args.torch_version, args.stage, args.network, not args.no_tex)
    target = save_report(report, args.output_dir)
    print(f'Saved: {target}\nSaved: {target.with_suffix(".txt")}')
    print(report['plan_error'] or report['torch_plan']['reason'])


if __name__ == '__main__':
    main()
