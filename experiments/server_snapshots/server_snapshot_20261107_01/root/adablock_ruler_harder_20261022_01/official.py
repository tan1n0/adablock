"""Load metric and prediction cleanup directly from the pinned official source."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE/'deps'))
import yaml

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def task_names():
    return json.loads((PACKAGE/'protocol.json').read_text())['tasks']

def sources():
    source=PACKAGE/'RULER/scripts'
    configs=yaml.safe_load((source/'synthetic.yaml').read_text())
    metrics=module(source/'eval/synthetic/constants.py','official_ruler_metrics')
    constants=module(source/'data/synthetic/constants.py','official_ruler_constants')
    # Extract the unchanged pure function without running CLI/dependency side effects.
    tree=ast.parse((source/'eval/evaluate.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='postprocess_pred')
    namespace={'re':re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 str(source/'eval/evaluate.py'),'exec'),namespace)
    return configs,metrics,constants,namespace['postprocess_pred']

_sources=None
def loaded():
    global _sources
    if _sources is None:_sources=sources()
    return _sources

def normalize_prediction(text,task):
    configs,_,_,postprocess=loaded()
    return postprocess(text,configs[task])

def metric_for(task):
    configs,metrics,_,_=loaded()
    return metrics.TASKS[configs[task]['task']]['metric_fn']

def limit_for(task):
    configs,_,constants,_=loaded()
    return constants.TASKS[configs[task]['task']]['tokens_to_generate']
