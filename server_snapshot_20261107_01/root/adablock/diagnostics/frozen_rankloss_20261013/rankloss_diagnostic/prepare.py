"""Validate prepared independent data; never launch collection or resplit data."""
import argparse
from pathlib import Path

from .data import check_independent_test, read_json


def resolve_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    for name in ('data', 'checkpoint', 'freeze_receipt', 'test_declaration',
                 'train_manifest', 'validation_manifest', 'reserved_test_manifest',
                 'attention_manifest', 'repo_root', 'training_config'):
        if config.get(name):
            candidate = Path(config[name])
            config[name] = str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    try:
        config = resolve_config(args.config)
        receipt = read_json(config['freeze_receipt'])
        manifest = read_json(Path(config['data']) / 'manifest.json')
        print(check_independent_test(config, manifest, receipt))
    except (ValueError, KeyError, FileNotFoundError) as error:
        p.exit(2, f'Independent test preparation stopped: {error}\n')


if __name__ == '__main__':
    main()
