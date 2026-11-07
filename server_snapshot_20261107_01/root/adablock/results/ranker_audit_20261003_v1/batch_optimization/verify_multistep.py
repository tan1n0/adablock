import sys,pathlib,json,copy,torch,time
ROOT=pathlib.Path('/root/adablock');sys.path.insert(0,str(ROOT))
from models.block_ranker import BlockRanker,BlockRankerConfig,to_device
from utils.ranking_data import RankingDataset
from utils.ranking_batch import BatchRankingDataset,pack_scenes,forward_packed,sample_batch_pairs,loss_from_pairs
from utils.ranking_loss import PairSamplingConfig,sample_pairs,weighted_rank_loss
from utils.ranking_eval import evaluate_ranker
from utils.ranking_batch_eval import evaluate_batched
torch.set_num_threads(1);torch.manual_seed(42)
out=pathlib.Path(__file__).parent
data=RankingDataset(ROOT/'data/ranker_cobs_validation16_20261003',cache_size=4)
newdata=BatchRankingDataset(ROOT/'data/ranker_cobs_validation16_20261003',cache_size=4)
meta=data.metadata
a=BlockRanker(BlockRankerConfig(meta['head_dim'],meta['num_layers'],meta['num_kv_heads'])).cuda()
b=copy.deepcopy(a);cfg=PairSamplingConfig()
o1=torch.optim.AdamW(a.parameters(),lr=2e-4,weight_decay=.01)
o2=torch.optim.AdamW(b.parameters(),lr=2e-4,weight_decay=.01)
g1=torch.Generator().manual_seed(43);g2=torch.Generator().manual_seed(43)
indices=torch.linspace(0,len(data)-1,256).long().tolist()
rows=[];start=time.perf_counter()
for begin in range(0,len(indices),8):
 batch=[data[i] for i in indices[begin:begin+8]]
 o1.zero_grad(set_to_none=True);o2.zero_grad(set_to_none=True)
 old=[a(to_device(s,'cuda:0')) for s in batch]
 p=pack_scenes(batch,'cuda:0');new=forward_packed(b,p)
 pa=[sample_pairs(old[i],s['teacher_mass'],s['fixed_mass'],cfg,g1).cpu() for i,s in enumerate(batch)]
 pb=sample_batch_pairs(new.detach().cpu(),batch,p['offsets'],cfg,g2)
 valid=[i for i,x in enumerate(pa) if x.numel()]
 la=torch.stack([weighted_rank_loss(old[i],batch[i]['teacher_mass'].cuda(),pa[i].cuda()) for i in valid]).mean() if valid else None
 lb,_=loss_from_pairs(new,batch,p['offsets'],pb)
 x=torch.cat(old).detach();y=new.detach();centered=float(((x-x.mean())-(y-y.mean())).abs().max())
 row={'step':begin//8+1,'pair_mismatch_scenes':sum(not torch.equal(i,j) for i,j in zip(pa,pb)),
      'rng_equal':torch.equal(g1.get_state(),g2.get_state()),'centered_score_max_abs':centered,
      'loss_abs':abs(float(la)-float(lb)) if la is not None else 0.}
 if la is not None:
  la.backward();lb.backward()
  torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True)
  torch.nn.utils.clip_grad_norm_(b.parameters(),1.,error_if_nonfinite=True)
  o1.step();o2.step()
 rows.append(row)
max_nonbias=max(float((x-y).abs().max()) for (name,x),(_,y) in zip(a.named_parameters(),b.named_parameters()) if name!='output.bias')
bias_delta=float((a.output.bias-b.output.bias).abs().max())
# Compare evaluator implementations using the SAME checkpoint/model.
old_eval=evaluate_ranker(b,data,'cuda:0',max_scenes=256)
new_eval=evaluate_batched(b,data,'cuda:0',max_scenes=256)
metric_errors=[]
for group,values in old_eval['groups'].items():
 for method,metrics in values['methods'].items():
  for key,value in metrics.items():
   error=abs(value-new_eval['groups'][group]['methods'][method][key])
   if error>1e-5:metric_errors.append({'group':group,'method':method,'key':key,'abs_error':error})
# Evaluate both independently updated models with the same evaluator.
ea=evaluate_batched(a,data,'cuda:0',max_scenes=256)
eb=evaluate_batched(b,data,'cuda:0',max_scenes=256)
coverage_diffs={k:abs(ea['groups']['overall']['methods']['learned'][f'coverage@{k}']-eb['groups']['overall']['methods']['learned'][f'coverage@{k}']) for k in ea['budgets']}
report={'steps':rows,'max_non_output_bias_parameter_difference':max_nonbias,
        'output_bias_difference_ranking_invariant':bias_delta,'evaluator_metric_errors_above_1e_5':metric_errors,
        'multistep_validation_coverage_differences':coverage_diffs,'seconds':time.perf_counter()-start}
assert all(r['rng_equal'] for r in rows),report
assert sum(r['pair_mismatch_scenes'] for r in rows)==0,report
assert max(r['loss_abs'] for r in rows)<1e-5,report
assert max_nonbias<1e-4,report
assert not metric_errors,report
assert max(coverage_diffs.values())<1e-5,report
(out/'multistep_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='steps'}),flush=True)
