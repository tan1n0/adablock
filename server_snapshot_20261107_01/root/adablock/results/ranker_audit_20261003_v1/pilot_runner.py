import subprocess,time,json,pathlib,os,sys
out=pathlib.Path(__file__).parent
cmd=['/root/miniconda3/envs/ada/bin/python','-u','scripts/build_ranking_dataset.py','--input-jsonl',str(out/'pilot_32k.jsonl'),'--output-dir',str(out/'pilot_data'),'--layers','12','20','27','--block-size','16','--sink-tokens','32','--local-window-blocks','32','--cobs-rank','4','--sample-stride','64','--max-length','32768','--max-docs','1','--local-files-only']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false')
(out/'pilot_command.json').write_text(json.dumps({'argv':cmd,'CUDA_VISIBLE_DEVICES':'0'},indent=2))
t=time.time(); peak=0
with (out/'pilot.log').open('x') as log:
 p=subprocess.Popen(cmd,cwd='/root/adablock',env=env,stdout=log,stderr=subprocess.STDOUT)
 (out/'pilot.pid').write_text(str(p.pid))
 while p.poll() is None:
  q=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  peak=max(peak,int(q.strip())); time.sleep(2)
 result={'exit_code':p.returncode,'seconds':time.time()-t,'peak_gpu_memory_mib_sampled_2s':peak}
 (out/'pilot_result.json').write_text(json.dumps(result,indent=2))
 sys.exit(p.returncode)
