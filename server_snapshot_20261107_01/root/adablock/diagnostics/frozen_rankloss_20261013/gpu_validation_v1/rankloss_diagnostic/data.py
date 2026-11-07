"""Manifest-only split checks and a local smoke reader of the audited v1 schema."""
import hashlib
import json
from bisect import bisect_right
from pathlib import Path

import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text('utf-8'))


def identities(manifest):
    result = set()
    for doc in manifest['documents']:
        if not doc.get('text_sha256') or not doc.get('components'):
            raise ValueError('Every document needs text_sha256 and original component identities')
        result.add('text:' + doc['text_sha256'])
        result.update('component:' + str(v) for v in doc['components'])
    return result


def check_independent_test(config, manifest, receipt):
    """Require explicit preexisting holdout evidence; never repartition a trained pool."""
    meta = manifest['metadata']
    if meta['layers'] != list(range(28)):
        raise ValueError('Formal test requires all 28 layers; old 3-layer test cannot be substituted')
    for doc_index in range(len(manifest['documents'])):
        entries = [s for s in manifest['shards'] if s['document_index'] == doc_index]
        if sorted(s['layer_id'] for s in entries) != list(range(28)) or any(s['heads'] != 4 for s in entries):
            raise ValueError('Each test document must have exactly 28 layer shards and 4 KV heads')
    declaration = read_json(config['test_declaration'])
    if (declaration.get('split') != 'independent_test'
            or declaration.get('used_for_training') is not False
            or declaration.get('used_for_checkpoint_selection') is not False):
        raise ValueError('Explicit independent test declaration required; selection val is not test')
    if declaration.get('test_manifest_sha256') != sha256(Path(config['data']) / 'manifest.json'):
        raise ValueError('Test declaration manifest hash mismatch')
    test_ids = identities(manifest)
    hashes = {}
    compatibility = ('schema', 'model_name', 'model_signature', 'block_size', 'sink_blocks',
                     'local_window_blocks', 'cobs_rank', 'layers', 'query_scaling', 'rope', 'gqa')
    for name in ('train', 'validation'):
        path = config[name + '_manifest']
        hashes[name] = sha256(path)
        if hashes[name] != receipt['manifest_hashes'][name]:
            raise ValueError(f'{name} manifest is not the frozen training/selection manifest')
        other = read_json(path)
        if any(meta.get(k) != other['metadata'].get(k) for k in compatibility):
            raise ValueError(f'{name}/test metadata mismatch')
        overlap = test_ids & identities(other)
        if overlap:
            raise ValueError(f'Test overlaps {name} by text/component identity: {sorted(overlap)[:5]}')
    reserved_path = config['reserved_test_manifest']
    if sha256(reserved_path) != receipt['reserved_test_manifest_sha256']:
        raise ValueError('Reserved test manifest differs from the preexisting audited holdout')
    if sha256(reserved_path) != declaration.get('reserved_manifest_sha256'):
        raise ValueError('Preexisting reserved test manifest hash mismatch')
    reserved = read_json(reserved_path)
    # Layer metadata can differ; document identity must remain exactly frozen.
    reserved_docs = {json.dumps(d, sort_keys=True) for d in reserved['documents']}
    if not all(json.dumps(d, sort_keys=True) in reserved_docs for d in manifest['documents']):
        raise ValueError('Test documents are not original preexisting heldout documents')
    if not declaration.get('preparation_note'):
        raise ValueError('Provide explicit all-layer preparation provenance')
    return dict(hashes, reserved_test=sha256(reserved_path), test=declaration['test_manifest_sha256'])


class LocalAuditDataset:
    """CPU smoke fallback; field mappings match captured RankingDataset exactly.

    Does not collect Q/K, teacher attention, or train a model. Production uses
    the repository RankingDataset directly.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / 'manifest.json')
        self.metadata = self.manifest['metadata']
        if self.metadata.get('schema') != 'adablock_cobs_ranking_v1':
            raise ValueError('Unsupported ranking schema')
        self.shards = self.manifest['shards']
        self.ends = []
        total = 0
        for entry in self.shards:
            total += entry['scenes']
            self.ends.append(total)
        self.cached = None

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.ends, index)
        entry = self.shards[shard_index]
        if self.cached is None or self.cached[0] != shard_index:
            self.cached = (shard_index, torch.load(self.directory / entry['file'], map_location='cpu', weights_only=True))
        shard = self.cached[1]
        offset = index - (self.ends[shard_index - 1] if shard_index else 0)
        record_index, head = divmod(offset, entry['heads'])
        record, digest = shard['records'][record_index], shard['digests']
        ids = record['block_ids']
        return {
            'query': record['query'][head], 'mean': digest['mean'][head, ids],
            'basis': digest['covariance_basis'][head, ids],
            'eigenvalues': digest['covariance_eigenvalues'][head, ids],
            'minimum': digest['minimum'][head, ids], 'maximum': digest['maximum'][head, ids],
            'counts': digest['counts'][ids], 'block_ids': ids,
            'teacher_mass': record['teacher_mass'][head], 'fixed_mass': record['fixed_mass'][head],
            'token_position': record['token_position'], 'layer_id': entry['layer_id'],
            'kv_head_id': head, 'document_index': entry['document_index'],
            'block_size': self.metadata['block_size'],
        }


def scene_key(s):
    return ':'.join(str(int(s[k])) for k in ('document_index', 'layer_id', 'token_position', 'kv_head_id'))
