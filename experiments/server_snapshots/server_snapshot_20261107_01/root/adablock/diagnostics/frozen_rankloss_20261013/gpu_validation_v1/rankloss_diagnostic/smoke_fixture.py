"""Make tiny synthetic CPU data/checkpoint; NEVER an independent test result."""
import argparse
import json
from pathlib import Path

import torch

from recovered.block_ranker import BlockRanker, BlockRankerConfig, save_ranker
from .data import sha256
from .run import json_write


def create_fixture(root, with_attention=True):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    data = root / 'data'
    data.mkdir(parents=True)
    torch.manual_seed(42)
    metadata = {'schema': 'adablock_cobs_ranking_v1', 'model_name': 'SYNTHETIC_NOT_QWEN_FORWARD',
                'model_signature': {'num_hidden_layers':28, 'num_attention_heads':28, 'num_key_value_heads':4},
                'block_size':16, 'sink_blocks':2, 'local_window_blocks':32, 'cobs_rank':4,
                'layers':[0], 'head_dim':128, 'query_scaling':'attention_scale_applied_once',
                'rope':'synthetic_post_rope_coordinates', 'gqa':7}
    documents, entries, sidecars = [], [], {}
    # Unequal records/document exercise both aggregation weights; candidates below/above64.
    for doc_index, (middle, records) in enumerate(((6, 1), (70, 2))):
        documents.append({'source':'synthetic', 'doc_id':f'smoke-{doc_index}',
                          'text_sha256':f'synthetic-{doc_index}', 'components':[f'synthetic::{doc_index}']})
        positions = [(middle+34+i)*16 for i in range(records)]
        length = positions[-1]+1
        keys = torch.randn(4, length, 128) * .3
        values = torch.randn(4, length, 128)
        blocks = (length + 15)//16
        digests = {'counts':torch.tensor([min(16, length-16*b) for b in range(blocks)], dtype=torch.float32)}
        chunks = []
        for b in range(blocks):
            k = keys[:,b*16:min((b+1)*16,length)]
            mean = k.mean(1)
            _, singular, vh = torch.linalg.svd(k-mean[:,None], full_matrices=False)
            basis = torch.zeros(4,4,128); eigen = torch.zeros(4,4)
            rank = min(4, k.shape[1])
            basis[:,:rank] = vh[:,:rank]
            eigen[:,:rank] = singular[:,:rank].square()/k.shape[1]
            chunks.append((mean, k.amin(1), k.amax(1), basis, eigen))
        for i, name in enumerate(('mean','minimum','maximum','covariance_basis','covariance_eigenvalues')):
            digests[name] = torch.stack([chunk[i] for chunk in chunks], dim=1)
        shard_records = []
        for t in positions:
            query = torch.randn(4,7,128) / (128**.5)
            attention = torch.softmax(torch.einsum('hgd,htd->hgt',query,keys[:,:t+1]), dim=-1, dtype=torch.float32)
            ids = torch.arange(2,t//16-32, dtype=torch.long)
            mass = torch.stack([attention[:,:,int(b)*16:(int(b)+1)*16].sum(-1).mean(-1) for b in ids],dim=-1)
            fixed = 1-mass.sum(-1)
            shard_records.append({'token_position':t, 'query':query, 'block_ids':ids,
                                  'teacher_mass':mass, 'fixed_mass':fixed})
            if with_attention:
                for head in range(4):
                    key = f'{doc_index}:0:{t}:{head}'
                    file = f'attention_{doc_index}_{t}_{head}.pt'
                    torch.save({'format':'adablock_attention_values_v1', 'scene_key':key,
                                'attention':attention[head], 'values':values[head,:t+1]},root/file)
                    sidecars[key] = {'file':file,'sha256':sha256(root/file)}
        file = f'doc_{doc_index:06d}_layer_000.pt'
        torch.save({'schema':metadata['schema'], 'layer_id':0,'digests':digests,'records':shard_records},data/file)
        entries.append({'file':file, 'document_index':doc_index,'layer_id':0,'records':records,'heads':4,'scenes':4*records})
    json_write(data/'manifest.json', {'metadata':metadata,'documents':documents,'shards':entries})
    model = BlockRanker(BlockRankerConfig(128,28,4)).float().eval()
    save_ranker(root/'synthetic_ranker.pt', model, metadata, {'step':0,'synthetic':True})
    json_write(root/'synthetic_receipt.json', {'synthetic':True,'step':0,'checkpoint_path':'synthetic_ranker.pt',
                                            'checkpoint_sha256':sha256(root/'synthetic_ranker.pt')})
    config = {'data':'data','checkpoint':'synthetic_ranker.pt','freeze_receipt':'synthetic_receipt.json'}
    if with_attention:
        json_write(root/'attention_index.json', {'format':'adablock_attention_index_v1',
                                               'ranking_manifest_sha256':sha256(data/'manifest.json'), 'scenes':sidecars})
        config['attention_manifest'] = 'attention_index.json'
    json_write(root/'config.json', config)
    return root/'config.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--without-attention', action='store_true')
    args = p.parse_args(); torch.set_num_threads(1)
    print(create_fixture(args.output, not args.without_attention))


if __name__ == '__main__':
    main()
