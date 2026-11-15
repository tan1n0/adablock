import pathlib,json,time,datetime
ROOT=pathlib.Path('/root/adablock');OUT=ROOT/'results/ranker_audit_20261003_v1/batch_optimization/ddp_fresh_run'
DOC=ROOT/'docs/ranker_full_training_20261003.md'
START='<!-- RANKER_STATUS_START -->';END='<!-- RANKER_STATUS_END -->'
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,json.JSONDecodeError):return None
while True:
    status=read(OUT/'formal_pipeline_status.json') or {}
    progress=read(ROOT/'checkpoints/ranker_ddp_fresh_global64_20261003/progress.json')
    history=read(ROOT/'checkpoints/ranker_ddp_fresh_global64_20261003/history.json') or []
    remote=datetime.datetime.now().astimezone().isoformat()
    lines=['','最近观察的远端时刻：'+remote+'（远端钟偏快约10分26秒；不是准确UTC）。',
           '','当前阶段：'+str(status.get('stage','unknown'))+'；流程PID：'+str(status.get('runner_pid'))+'；子进程PID：'+str(status.get('child_pid','无/已退出'))+'。']
    if progress:
        lines+=['','最近训练进度：epoch '+str(progress['epoch'])+'，'+str(progress['seen'])+'/'+str(progress['total_scenes'])+' 场景；loss='+str(progress['train_loss'])+'；速度='+str(progress['scenes_per_second'])+'场景/秒。']
    lines+=['','已完成 epoch 数：'+str(len(history))+'。']
    if history:
        lines+=['','| Epoch | Train loss | 验证平均 coverage@K | 训练秒数 |','| --- | ---: | ---: | ---: |']
        lines += ['| '+str(r['epoch'])+' | '+format(r['train_loss'],'.6f')+' | '+format(r['mean_validation_coverage'],'.6f')+' | '+format(r.get('training_seconds',0),'.1f')+' |' for r in history]
    if status.get('error'):lines+=['','阻塞：'+status['error']]
    if status.get('stage')=='completed':
        result=read(OUT/'heldout_test_comparison.json')
        if result:
            lines+=['','独立测试已完成；场景数 '+str(result['scenes'])+'；选中epoch '+str(result['checkpoint_epoch'])+'。','',
                    '| 方法 | Coverage@8 | Coverage@32 | Coverage@128 | Required k |',
                    '| --- | ---: | ---: | ---: | ---: |']
            for name,m in result['groups']['overall']['methods'].items():
                lines+=['| '+name+' | '+format(m['coverage@8'],'.6f')+' | '+format(m['coverage@32'],'.6f')+' | '+format(m['coverage@128'],'.6f')+' | '+format(m['required_k'],'.2f')+' |']
    text=DOC.read_text()
    if START not in text or END not in text:raise RuntimeError('Documentation markers changed; preserving user content.')
    before,after=text.split(START,1);_,after=after.split(END,1)
    tmp=DOC.with_suffix('.tmp');tmp.write_text(before+START+'\n'.join(lines)+'\n'+END+after);tmp.replace(DOC)
    if status.get('stage') in ['completed','blocked']:break
    time.sleep(60)
