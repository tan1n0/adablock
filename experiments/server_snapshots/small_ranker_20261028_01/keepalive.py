"""Idle GPU pulses. Yield whenever training/evaluation owns a device."""
import os,sys,time,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if len(sys.argv)>1 and sys.argv[1]=='pulse':
    import torch
    x=torch.randn((1024,1024),device='cuda',dtype=torch.float16)
    end=time.monotonic()+1
    while time.monotonic()<end:
        for _ in range(8):y=x@x
        torch.cuda.synchronize();time.sleep(.02)
else:
    while not (ROOT/'STOP_KEEPALIVE').exists():
        if not (ROOT/'PAUSE_KEEPALIVE').exists():
            devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
            jobs=[]
            for row in devices.splitlines():
                index,uuid=[s.strip() for s in row.split(',')]
                if uuid not in busy:
                    jobs.append(subprocess.Popen([sys.executable,__file__,'pulse'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=index),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
            for job in jobs:job.wait()
            (ROOT/'keepalive_status.json').write_text(json.dumps({'watchdog_pid':os.getpid(),'last_check_unix':time.time(),'idle_devices_pulsed':len(jobs)}))
        time.sleep(15)
