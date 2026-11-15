"""Freeze the S/A/B protocol and check that richer data reproduce old supervision."""
import json
import torch
import core as c


def main():
    torch.set_num_threads(4)
    files=sorted((c.ROOT/'raw_qa').glob('*.pt'))
    assert len(files)==60
    audit={};hashes={}
    for file in files:
        p=torch.load(file,map_location='cpu',weights_only=True)
        audit[file.name]=p['verification'];hashes[file.name]=c.sha(file)
    c.dump(c.ROOT/'collection_verification.json',audit)
    c.dump(c.ROOT/'raw_data_hashes.json',hashes)
    ranker,budget,_=c.initial_models('cuda:0')
    groups=c.qa_groups('validation')
    d=c.supervision(ranker,groups,'cuda:0')
    old=torch.load(c.QA/'budget_data/validation.pt',map_location='cpu',weights_only=True)
    lookup={tuple([int(ids[1]),int(ids[2]),int(ids[3]),int(ids[4]),int(ids[5]),int(t)]):i
            for i,(ids,t) in enumerate(zip(old['identities'],old['trajectory']))}
    assert len(lookup)==len(old['features'])==len(d['features'])
    indices=torch.tensor([lookup[tuple(ids.tolist())] for ids in d['identities']])
    feature_error=float((d['features']-old['features'][indices]).abs().max())
    coverage_error=float((d['coverage']-old['coverage'][indices]).abs().max())
    assert feature_error<.005 and coverage_error<2e-5,(feature_error,coverage_error)
    c.dump(c.ROOT/'refresh_parity.json',dict(feature_max_error=feature_error,coverage_max_error=coverage_error,
                                          validation_scenes=len(indices),all_identities_match=True))
    protocol=dict(config=c.CONFIG,branches=dict(S='rank0,rank1,budget0,budget1',A='rank0,budget0,rank1,budget1',
                                               B='same as A plus actual-K boundary loss'),
                  initial_ranker=str(c.PILOT/'ranker/best.pt'),initial_budget=str(c.PREVIOUS/'rho90/best.pt'),
                  initial_ranker_sha256=c.sha(c.PILOT/'ranker/best.pt'),initial_budget_sha256=c.sha(c.PREVIOUS/'rho90/best.pt'),
                  rank_data='original pilot rank_train, 12 documents, 4 evenly-spaced positions per document/layer, all 8 KV heads',
                  budget_data='same original 48 train and 12 held-out WikiHop QA, same full/old-adaptive trajectories',
                  raw_data_manifest_sha256=c.sha(c.ROOT/'raw_data_hashes.json'),
                  qa_selection_sha256=c.sha(c.ROOT/'qa_selected.json'),
                  qa='same frozen 32 HotpotQA TRAIN questions from previous rho comparison; no new QA selection',
                  policy='q=.95 remains fixed; rho=.90 for all label generation and rank positives',
                  common_objective='original CE only for every budget update; no changed cost/quality penalties',
                  fairness='same initial models, data, normalization, rank updates and budget epochs; bounded pair count; report actual pairs and time',
                  teacher='full attention offline only; no teacher access in deployed selection',
                  boundary='hard K and hard top-K detached; selected low-mass versus unselected high-mass blocks; <=32 extra pairs; weight 1',
                  constraints='No coverage-target changes, hyperparameter sweeps or QA-based checkpoint selection',
                  gpu_collection='4 separate single-GPU workers; full Llama unloaded before small-model training')
    assert c.sha(c.ROOT/'qa_selected.json')==c.sha(c.PREVIOUS/'qa_selected.json')
    c.dump(c.ROOT/'protocol.json',protocol)
    print(json.dumps(dict(stage='prepared',feature_error=feature_error,coverage_error=coverage_error)),flush=True)


if __name__=='__main__':main()
