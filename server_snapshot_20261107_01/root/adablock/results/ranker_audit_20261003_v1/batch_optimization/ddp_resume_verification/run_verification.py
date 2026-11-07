import pathlib,subprocess,json,os,torch
root=pathlib.Path('/root/adablock');out=pathlib.Path(__file__).parent
base=['/root/miniconda3/envs/ada/bin/python','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4','scripts/train_block_ranker_ddp.py','--train-dir',str(out/'train'),'--val-dir',str(out/'validation'),'--epochs','2','--local-batch','2','--checkpoint-every-scenes','16','--log-every-batches','2','--trace-indices']
for tag,extra in [('continuous',[]),('interrupted',['--max-run-scenes','32']),('resumed',['--resume',str(out/'interrupted/resume_latest.pt')])]:
 cmd=base+['--output-dir',str(out/tag)]+extra
 with (out/(tag+'.log')).open('x') as log:
  p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1'))
  (out/'status.json').write_text(json.dumps({'stage':tag,'pid':p.pid},indent=2))
  rc=p.wait()
 if rc:
  (out/'status.json').write_text(json.dumps({'stage':'failed','tag':tag,'exit_code':rc},indent=2))
  raise RuntimeError(tag)
a=torch.load(out/'continuous/resume_latest.pt',weights_only=True,map_location='cpu')
b=torch.load(out/'resumed/resume_latest.pt',weights_only=True,map_location='cpu')
tr1=json.loads((out/'continuous/trace_indices.json').read_text());tr2=json.loads((out/'resumed/trace_indices.json').read_text())
assert tr1==tr2 and len(tr1)==120
assert all(len([i for e,i in tr1 if e==epoch])==len({i for e,i in tr1 if e==epoch})==60 for epoch in [1,2])
diff={k:float((a['model_state_dict'][k]-b['model_state_dict'][k]).abs().max()) for k in a['model_state_dict']}
nonbias=max(v for k,v in diff.items() if k!='output.bias')
assert nonbias<2e-5,diff
for ra,rb in zip(a['rank_rng_states'],b['rank_rng_states']):
 assert torch.equal(ra['pair'],rb['pair'])
 assert torch.equal(ra['torch_cpu'],rb['torch_cpu'])
 assert torch.equal(ra['torch_cuda'],rb['torch_cuda'])
 assert ra['python']==rb['python']
assert torch.equal(a['epoch_start_sampler_rng'],b['epoch_start_sampler_rng'])
steps_a=[float(v['step']) for v in a['optimizer_state_dict']['state'].values()]
steps_b=[float(v['step']) for v in b['optimizer_state_dict']['state'].values()]
assert steps_a==steps_b
moment_diff=max(float((a['optimizer_state_dict']['state'][k][m]-b['optimizer_state_dict']['state'][k][m]).abs().max()) for k in a['optimizer_state_dict']['state'] for m in ['exp_avg','exp_avg_sq'])
assert moment_diff<2e-5,moment_diff
validation_diff=max(abs(x['mean_validation_coverage']-y['mean_validation_coverage']) for x,y in zip(a['history'],b['history']))
assert validation_diff<2e-5,validation_diff
result={'passed':True,'epochs':2,'scenes_per_epoch':60,'total_indices':120,'trace_exactly_equal':True,'no_skipped_or_repeated_scene_per_epoch':True,'all_rank_rng_exactly_equal':True,'sampler_epoch_start_rng_exactly_equal':True,'optimizer_steps_exactly_equal':True,'max_non_output_bias_parameter_difference':nonbias,'output_bias_difference':diff['output.bias'],'max_optimizer_moment_difference':moment_diff,'max_validation_mean_coverage_difference':validation_diff,'checkpoint_next_epoch':b['epoch'],'checkpoint_next_scene_offset':b['next_scene_offset']}
(out/'result.json').write_text(json.dumps(result,indent=2))
(out/'status.json').write_text(json.dumps({'stage':'completed'},indent=2))
print(json.dumps(result),flush=True)
