"""Remote detached launch after upload; never launch a duplicate driver."""
import json
import pathlib
import subprocess
import sys

root=pathlib.Path('/root/adablock_ruler_harder_20261022_01')
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:command=(proc/'cmdline').read_bytes()
    except (OSError,ProcessLookupError):continue
    if str(root/'driver.py').encode() in command:
        print(json.dumps(dict(stage='already_running',pid=int(proc.name))));sys.exit(0)
subprocess.run([sys.executable,'-m','py_compile',*[str(p) for p in root.glob('*.py')]],check=True)
with (root/'driver.log').open('a') as log:
    process=subprocess.Popen([sys.executable,'-u',str(root/'driver.py')],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(stage='launched',pid=process.pid)))
