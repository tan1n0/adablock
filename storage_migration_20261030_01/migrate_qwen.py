"""Move the authorized Qwen dataset in verified, resumable document batches."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

SOURCE = Path('/root/adablock/data/ranker_cobs_train_all28_20261004')
DEST = Path('/root/autodl-tmp/adablock_data/ranker_cobs_train_all28_20261004')
CONTROL = Path('/root/autodl-tmp/adablock_data/_migration_qwen_all28_20261030_01')


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path, data):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    syncdir(path.parent)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(path):
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode), str(path)
    return [s.st_size, s.st_mtime_ns, s.st_ino]


def copy_verified(src, dst, expected):
    assert fingerprint(src) == expected, 'Source changed: ' + str(src)
    tmp = dst.with_name(dst.name + '.migration_partial')
    assert not tmp.is_symlink() and not dst.is_symlink()
    h = hashlib.sha256()
    with src.open('rb') as inp, tmp.open('wb') as out:
        for b in iter(lambda: inp.read(4 * 1024 * 1024), b''):
            h.update(b)
            out.write(b)
        out.flush()
        shutil.copystat(src, tmp)
        owner = src.stat()
        os.chown(tmp, owner.st_uid, owner.st_gid)
    assert fingerprint(src) == expected, 'Source changed while reading'
    assert tmp.stat().st_size == expected[0] and sha(tmp) == h.hexdigest(), 'Copy checksum mismatch'
    os.replace(tmp, dst)
    return {'size': expected[0], 'sha256': h.hexdigest()}


def syncfiles(paths):
    # Flush a whole document after copying, avoiding a read/write disk seek per file.
    for path in paths:
        with path.open('rb') as f:
            os.fsync(f.fileno())


def main(limit):
    assert SOURCE.parent.resolve() == Path('/root/adablock/data')
    assert DEST.parent.resolve() == Path('/root/autodl-tmp/adablock_data')
    assert os.path.ismount('/root/autodl-tmp')
    CONTROL.mkdir(exist_ok=True)
    lock = (CONTROL / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (CONTROL / 'verified').mkdir(exist_ok=True)
    inventory_path = CONTROL / 'inventory.json'
    if not inventory_path.exists():
        assert SOURCE.is_dir() and not SOURCE.is_symlink() and not DEST.exists()
        assert {p.name for p in SOURCE.iterdir()} == {'manifest.json', 'docs'}
        docs = {}
        for d in sorted((SOURCE / 'docs').iterdir()):
            assert d.is_dir() and not d.is_symlink() and d.name.startswith('doc_')
            docs[d.name] = {p.name: fingerprint(p) for p in sorted(d.iterdir())}
            assert len(docs[d.name]) == 30
        inventory = {'source': str(SOURCE), 'destination': str(DEST), 'documents': docs,
                     'manifest': fingerprint(SOURCE / 'manifest.json')}
        assert len(docs) == 1751
        save(inventory_path, inventory)
    inventory = json.loads(inventory_path.read_text())
    assert inventory['source'] == str(SOURCE) and inventory['destination'] == str(DEST)
    if SOURCE.is_symlink():
        assert SOURCE.resolve() == DEST
        assert (CONTROL / 'manifest_verified.json').exists()
        for name, files in inventory['documents'].items():
            assert (CONTROL / 'verified' / (name + '.json')).exists()
            for filename, expected in files.items():
                assert (DEST / 'docs' / name / filename).stat().st_size == expected[0]
        print('Already migrated; source link and all file sizes verified.', flush=True)
        return
    DEST.mkdir(exist_ok=True)
    assert not DEST.is_symlink()
    (DEST / 'docs').mkdir(exist_ok=True)
    docs = inventory['documents']
    total = sum(v[0] for files in docs.values() for v in files.values())
    initial = sum(sum(v[0] for v in files.values()) for name, files in docs.items()
                  if (CONTROL / 'verified' / (name + '.json')).exists())
    started = time.monotonic()
    done = 0
    count = 0
    migrated_this_run = 0

    def status(state, current=None):
        elapsed = time.monotonic() - started
        rate = max(0, done - initial) / max(elapsed, .001)
        data = {'state': state, 'pid': os.getpid(), 'source': str(SOURCE), 'destination': str(DEST),
                'documents_done': count, 'documents_total': len(docs), 'bytes_done': done,
                'bytes_total': total, 'percent': round(done / total * 100, 3),
                'MiB_per_second': round(rate / 2**20, 2),
                'eta_seconds': round((total - done) / rate) if rate else None,
                'current_document': current, 'timestamp': time.time()}
        save(CONTROL / 'status.json', data)
        print(json.dumps(data), flush=True)

    for name, files in docs.items():
        src = SOURCE / 'docs' / name
        target = DEST / 'docs' / name
        old = SOURCE / 'docs' / ('.' + name + '.migration_original')
        record_path = CONTROL / 'verified' / (name + '.json')
        if not record_path.exists():
            status('copying', name)
            assert not src.is_symlink() and src.is_dir() and not old.exists()
            target.mkdir(exist_ok=True)
            assert not target.is_symlink()
            records = {filename: copy_verified(src / filename, target / filename, expected)
                       for filename, expected in files.items()}
            syncfiles(target / filename for filename in files)
            shutil.copystat(src, target)
            syncdir(target)
            syncdir(target.parent)
            save(record_path, records)
            migrated_this_run += 1
        records = json.loads(record_path.read_text())
        assert set(records) == set(files) == {p.name for p in target.iterdir()}
        for filename, expected in files.items():
            p = target / filename
            assert p.is_file() and not p.is_symlink() and p.stat().st_size == expected[0]
        if not src.is_symlink():
            if not old.exists():
                assert {p.name for p in src.iterdir()} == set(files)
                for filename, expected in files.items():
                    assert fingerprint(src / filename) == expected
                os.rename(src, old)
                syncdir(old.parent)
            assert not src.exists()
            src.symlink_to(target, target_is_directory=True)
            syncdir(src.parent)
        assert src.resolve() == target.resolve()
        if old.exists():
            assert not old.is_symlink() and old.parent == SOURCE / 'docs'
            for p in list(old.iterdir()):
                assert p.name in files and fingerprint(p) == files[p.name]
                p.unlink()
            old.rmdir()
            syncdir(old.parent)
        done += sum(v[0] for v in files.values())
        count += 1
        status('moving')
        if limit and migrated_this_run >= limit:
            status('pilot_complete')
            return

    if not SOURCE.is_symlink():
        record = copy_verified(SOURCE / 'manifest.json', DEST / 'manifest.json', inventory['manifest'])
        syncfiles([DEST / 'manifest.json'])
        save(CONTROL / 'manifest_verified.json', record)
        syncdir(DEST)
        manifest = json.loads((DEST / 'manifest.json').read_text())
        assert len(manifest['shards']) == 49028
        for entry in manifest['shards']:
            assert (DEST / entry['file']).is_file(), entry['file']
        assert len(list((SOURCE / 'docs').iterdir())) == len(docs)
        leftovers = SOURCE.with_name(SOURCE.name + '.migration_links_20261030_01')
        assert not leftovers.exists()
        os.rename(SOURCE, leftovers)
        try:
            SOURCE.symlink_to(DEST, target_is_directory=True)
            syncdir(SOURCE.parent)
        except BaseException:
            if not os.path.lexists(SOURCE):
                os.rename(leftovers, SOURCE)
            raise
        for p in (leftovers / 'docs').iterdir():
            assert p.is_symlink() and p.resolve() == DEST / 'docs' / p.name
            p.unlink()
        (leftovers / 'docs').rmdir()
        assert sha(leftovers / 'manifest.json') == record['sha256']
        (leftovers / 'manifest.json').unlink()
        leftovers.rmdir()
        syncdir(SOURCE.parent)
    assert SOURCE.is_symlink() and SOURCE.resolve() == DEST
    status('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-docs', type=int, default=0)
    args = parser.parse_args()
    try:
        main(args.limit_docs)
    except BaseException as error:
        if CONTROL.exists():
            save(CONTROL / 'failure.json', {'error': repr(error), 'time': time.time(), 'pid': os.getpid()})
        raise
