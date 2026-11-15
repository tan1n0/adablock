"""Reproduce original generated trajectories and retain refreshable raw teacher scenes."""
import gc
import importlib.util
import json
import sys

import torch
import core as c

spec=importlib.util.spec_from_file_location('original_qa_engine',c.QA/'engine.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)


class RichCollector(engine.Collector):
    def install(self):
        self.raw=[]
        super().install()

    def attention(self,query,keys,values,indices,valid,scale):
        if self.probe is not None:
            layer,ids,scores,features=self.probe
            h=keys.shape[1]
            grouped=query[0,:,0].float().reshape(h,-1,query.shape[-1])
            mass=torch.einsum('hgd,htd->hgt',grouped,keys[0].float()).mul(scale).softmax(-1).mean(1)
            blocks=torch.nn.functional.pad(mass,(0,(-mass.shape[-1])%16)).reshape(h,-1,16).sum(-1)
            candidate=blocks[:,ids].cpu()
            self.raw.append(dict(query=(grouped*scale).cpu(),teacher_mass=candidate,
                                 fixed_mass=1-candidate.sum(-1),block_ids=ids.cpu(),
                                 token_position=self.position,layer_id=layer))
        return super().attention(query,keys,values,indices,valid,scale)


@torch.inference_mode()
def main(shard):
    c.run.AdaptiveLlama=RichCollector
    model,tok,ranker,budget,eos,_=c.run.load()
    rows=[r for r in json.loads((c.QA/'selected.json').read_text()) if r['split'] in ['budget_train','validation']]
    c.run.prefill(model,rows[0]['input_ids'][:128])
    for i,row in enumerate(rows):
        if i%4!=shard:continue
        target=c.ROOT/'raw_qa'/f"{row['split']}_{row['document']:03d}.pt"
        assert not target.exists()
        original=torch.load(c.QA/'collected'/f"{row['split']}_{row['document']}.pt",map_location='cpu',weights_only=True)
        pairs,logits,_=c.run.prefill(model,row['input_ids'])
        digests=c.rt.build_prefix_digests(pairs)
        result=dict(document=row,rows={},digests={},verification={})
        cursor=0
        for trajectory,mode in [('full','all'),('old_adaptive','adaptive')]:
            output=c.run.decode(model,pairs,logits,mode,ranker,budget,digests,eos,quantile=.95,capture=True)
            controller=output.pop('_controller')
            assert output['ids']==original['generations'][trajectory]['ids'],(row['doc_id'],trajectory)
            old=[r for r in original['records'] if r['trajectory']==trajectory]
            assert len(old)==len(controller.records)==len(controller.raw)*8
            errors=dict(features=0.,coverage=0.)
            for a,b in zip(controller.records,old):
                errors['features']=max(errors['features'],float((a['features']-b['features']).abs().max()))
                errors['coverage']=max(errors['coverage'],float((a['coverage']-b['coverage']).abs().max()))
            assert errors['features']<.005 and errors['coverage']<2e-5,errors
            result['rows'][trajectory]=controller.raw
            result['digests'][trajectory]=[{k:v.cpu() for k,v in d.items()} for d in controller.digests]
            result['verification'][trajectory]=dict(ids_identical=True,scenes=len(old),max_errors=errors)
            del controller,output
        torch.save(result,target)
        print(json.dumps(dict(split=row['split'],document=row['document'],verification=result['verification'])),flush=True)
        del pairs,logits,digests,result,original
        gc.collect();torch.cuda.empty_cache()
    print('COMPLETE',flush=True)


if __name__=='__main__':main(int(sys.argv[1]))
