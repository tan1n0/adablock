"""Local loopback progress view; refresh SSH snapshot every 30 seconds."""
import json,time,threading,subprocess,os
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
ROOT=Path(__file__).resolve().parent
REMOTE='/root/adablock_small_ranker_20261028_01'
latest={'stage':'connecting'}
lock=threading.Lock()
def poll():
    global latest
    while True:
        try:
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','gpu-027-llm',
                '/root/miniconda3/envs/ada/bin/python '+REMOTE+'/progress_snapshot.py'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=25,creationflags=subprocess.CREATE_NO_WINDOW)
            if p.returncode:raise RuntimeError(p.stderr.strip()[-500:])
            data=json.loads(p.stdout)
            with lock:latest=data
            (ROOT/'progress_live.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            with (ROOT/'progress_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')
            if data['stage']=='complete':
                result=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=10',
                    'gpu-027-llm:'+REMOTE+'/report.txt','gpu-027-llm:'+REMOTE+'/comparison.json',
                    'gpu-027-llm:'+REMOTE+'/verification.json',str(ROOT)],capture_output=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode:raise RuntimeError('评测已完成，汇总下载暂未成功；稍后重试。')
                return
            if data['stage']=='failed':return
        except Exception as exc:
            with lock:latest=dict(latest,connection_error=str(exc),connection_checked_unix=time.time())
            (ROOT/'progress_live.json').write_text(json.dumps(latest,ensure_ascii=False,indent=2),encoding='utf-8')
        time.sleep(30)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=self.path.split('?')[0]
        if path=='/progress.json':
            with lock:body=json.dumps(latest,ensure_ascii=False).encode()
            kind='application/json; charset=utf-8'
        elif path=='/report.txt' and (ROOT/'report.txt').exists():body=(ROOT/'report.txt').read_bytes();kind='text/plain; charset=utf-8'
        elif path=='/':body=(ROOT/'progress.html').read_bytes();kind='text/html; charset=utf-8'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
(ROOT/'live_monitor_info.json').write_text(json.dumps({'pid':os.getpid(),'url':f'http://127.0.0.1:{server.server_port}/','refresh_seconds':30},indent=2),encoding='utf-8')
threading.Thread(target=poll,daemon=True).start()
server.serve_forever()
