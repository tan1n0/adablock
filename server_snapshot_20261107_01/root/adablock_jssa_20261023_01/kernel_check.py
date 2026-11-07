import ast, json, sys, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'kernel_adapter'))
tree=ast.parse((ROOT/'kernel_adapter/test_jssa_decode.py').read_text())
tree.body=[n for n in tree.body if not (isinstance(n,ast.Import) and any(a.name=='pytest' for a in n.names))]
for n in tree.body:
    if isinstance(n,ast.FunctionDef):n.decorator_list=[]
ns={};exec(compile(tree,'official_kernel_tests','exec'),ns)
report={}
try:
    ns['test_fp8_contig_score_matches_bf16'](128,32)
    report['fp8_score']='passed'
except Exception:
    report['fp8_score']='failed'
    report['fp8_error']=traceback.format_exc()
for group in [1,4]:
    ns['test_indexed_attend_matches_reference'](group,1,128)
    report[f'indexed_bf16_G{group}']='passed'
(ROOT/'kernel_check.json').write_text(json.dumps(report,indent=2))
print(report,flush=True)
