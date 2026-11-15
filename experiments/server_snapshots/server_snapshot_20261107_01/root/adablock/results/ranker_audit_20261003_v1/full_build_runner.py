import subprocess,time,json,pathlib,datetime,sys,hashlib
out=pathlib.Path(__file__).parent
spec=json.loads((out/'full_build_command.json').read_text())
root=pathlib.Path(spec['cwd'])
snapshot={}
for folder in ['scripts','utils','models']:
 for p in (root/folder).glob('*.py'): snapshot[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'source_sha256_at_full_launch.json').write_text(json.dumps(snapshot,indent=2))
start=time.time()
with (out/'full_build.log').open('x') as log:
 p=subprocess.Popen(spec['argv'],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 status={'state':'running','pid':p.pid,'runner_pid':__import__('os').getpid(),'remote_started_at':datetime.datetime.now().astimezone().isoformat(),'argv':spec['argv']}
 (out/'full_build_status.json').write_text(json.dumps(status,indent=2))
 (out/'full_build.pid').write_text(str(p.pid))
 rc=p.wait()
 status.update(state='completed' if rc==0 else 'failed',exit_code=rc,seconds=time.time()-start,remote_ended_at=datetime.datetime.now().astimezone().isoformat())
 (out/'full_build_status.json').write_text(json.dumps(status,indent=2))
 (out/('FULL_BUILD_COMPLETE' if rc==0 else 'FULL_BUILD_FAILED')).write_text(str(rc))
 sys.exit(rc)
