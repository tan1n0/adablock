import diagnose as d
import guarded_controller
import json,time,gc
import torch

class Intervention(d.e.gpu.Controller):
    def choose(self,layer,query,scale):
        indices,valid=super().choose(layer,query,scale)
        event=self.event
        if self.position==event['position'] and layer==event['layer']:
            for head in event['heads']:
                h=head['head']
                chosen=indices[h,valid[h]]
                blocks=set((chosen//16).tolist())
                blocks.difference_update(head['gained']);blocks.update(head['lost'])
                tokens=[t for b in sorted(blocks) for t in range(b*16,min((b+1)*16,self.position+1))]
                assert len(tokens)==int(valid[h].sum())
                indices[h,valid[h]]=torch.tensor(tokens,device=query.device)
        return indices,valid

@torch.inference_mode()
def main():
    row=next(r for r in json.loads((d.PREVIOUS/'selected.json').read_text()) if r['task']=='cwe' and r['length']==8192)
    model,tok,models,eos=d.e.load();rank,budget=models['B']
    p,l,_=d.e.prefill(model,row['input_ids'])
    Intervention.event=json.loads((d.ROOT/'trace_analysis.json').read_text())['first_selection_difference']
    report={}
    for name,cls in [('single_intervention',Intervention),('guarded',guarded_controller.Controller)]:
        digests,seconds=d.build(p,'gram')
        cls.ratio=.25;d.e.c.run.AdaptiveLlama=cls;d.e.c.run.DynamicCache=d.e.BufferCache
        d.e.baseline.compact_digest=d.function('gram')
        result=d.e.c.run.decode(model,p,l,'adaptive',rank,budget,digests,eos,max_new=row['max_new'],quantile=.95)
        prediction=d.e.official.normalize_prediction(tok.decode(result['ids'],skip_special_tokens=True),row['task'])
        report[name]=dict(ids=result['ids'],prediction=prediction,score=d.e.official.metric_for(row['task'])([prediction],[row['answers']]),
                          index_seconds=seconds,decode_seconds=result['decode_seconds'],audit=result['audit'])
        d.e.dump('guard_probe.json',report);print(name,json.dumps(report[name]),flush=True)
        del digests,result;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
