Qwen all-layer training data migration, 2026-10-30

Authorized operation: move only ranker_cobs_train_all28_20261004 to the data disk.
Source: /root/adablock/data/ranker_cobs_train_all28_20261004
Destination: /root/autodl-tmp/adablock_data/ranker_cobs_train_all28_20261004
Remote status and checksums:
  /root/autodl-tmp/adablock_data/_migration_qwen_all28_20261030_01/
Remote worker:
  /root/autodl-tmp/adablock_data/migrate_qwen_20261030_01.py
Remote log:
  /root/autodl-tmp/adablock_data/migration_qwen_20261030_01.log
Full-run PID at launch: 1925 (server restart may invalidate this PID).

The original inventory is backed up here. There are 1,751 document folders,
each with 28 tensor shards plus two JSON files, and one root manifest.
Each document is copied, independently SHA-256 checked against its source
stream, and flushed to disk before the source files are removed. Verified
hashes are persisted per document. The original document path becomes a
symbolic link. On completion the dataset root becomes a single symbolic link.

Four-document pilot: 120 files / 816,620,500 bytes migrated; a separate
read-back SHA-256 audit passed for all 120 files and all original paths worked.
The full migration was subsequently launched independently of SSH.
Migration is complete and independently audited; see verification_final.json.
All 1,751 documents / 52,531 files are present. SHA-256 was checked for every
file during migration. Post-migration verification checked every file size,
mtime and original-path resolution, the manifest hash, and 16 tensor hashes
sampled across eight documents. The full run took 12,077.6 seconds.
System filesystem visible file usage fell from 807.956 GiB to 290.986 GiB.

No model weights, other datasets, experiment results, or training settings are
changed. Both mounts share the host's backing storage, so this moves files out
of the container system directory without reducing total host disk usage.
