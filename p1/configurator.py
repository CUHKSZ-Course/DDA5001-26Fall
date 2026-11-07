"""Keep nanoGPT's interface: train.py config/file.py --key=value."""
from ast import literal_eval
from pathlib import Path


def load_config(defaults, arguments):
    config = dict(defaults)
    for argument in arguments:
        if argument.startswith('--'):
            key, value = argument[2:].split('=', 1)
            if key not in defaults:
                raise ValueError(f'Unknown setting: {key}')
            try:
                value = literal_eval(value)
            except (SyntaxError, ValueError):
                pass
            if isinstance(defaults[key], float) and type(value) is int:
                value = float(value)
            if type(value) is not type(defaults[key]):
                raise TypeError(f'{key} requires {type(defaults[key]).__name__}')
            config[key] = value
        else:
            values = {}
            exec(compile(Path(argument).read_text(), argument, 'exec'), {}, values)
            unknown = set(values) - set(defaults)
            if unknown:
                raise ValueError(f'Unknown config settings: {unknown}')
            config.update(values)
    return config
