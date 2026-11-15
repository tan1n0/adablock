"""Restore byte-verified existing ranker for local CPU smoke only."""
import hashlib
import json
from pathlib import Path


def recover(evidence_dir, output):
    evidence_dir, output = Path(evidence_dir), Path(output)
    network = json.loads((evidence_dir / 'rankloss_network_source_evidence.json').read_text('utf-8'))
    item = network['files']['models/block_ranker.py']
    source = '\n'.join(line.split(': ', 1)[1] for line in item['numbered_source'].splitlines()) + '\n'
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != item['frozen_sha256']:
        raise ValueError('Recovered source hash does not match audited repository source')
    target = output / 'block_ranker.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_bytes(source.encode())
    training = json.loads((evidence_dir / 'oct5_training_evidence.json').read_text('utf-8'))
    audit = training['FINAL_AUDIT.json']
    receipt = {
        'captured_utc': network.get('captured_utc', '2026-10-10 (local read-only audit)'),
        'evidence_source': str(evidence_dir / 'oct5_training_evidence.json'),
        'source_config_path': audit['deliverables']['training_config'],
        'source_config': training['run_config.json'],
        'checkpoint_path': audit['deliverables']['model'],
        'checkpoint_sha256': training['small_checkpoint_recheck']['best_sha256'],
        'step': audit['best_step'],
        'manifest_hashes': training['plan_audit.json']['manifest_hashes'],
        'reserved_test_manifest_path': '/root/adablock/data/ranker_cobs_test16_ddp_fresh_20261003/manifest.json',
        'reserved_test_manifest_sha256': '370f3ad1a728b647a0fa982b21e9ebacc4d65be7790a57e77aa418dc33a49ef6',
        'ranker_source_sha256': digest,
    }
    (output / 'freeze_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'recovered_source': str(target), 'sha256': digest, 'receipt': str(output / 'freeze_receipt.json')}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    recover(args.evidence_dir, args.output)
