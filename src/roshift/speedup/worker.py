"""Latency orchestration; one model/method per child, no competing GPU jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys

from .runtime import ENGINE, dump


def child(request, **changes):
    from ..accuracy.execution import run_child
    value = dict(request, **changes)
    path = Path(request['data_dir'])/'requests'/('speedup_child_'+__import__('uuid').uuid4().hex+'.json')
    dump(path, value)
    env = dict(os.environ, ROSHIFT_SPEED_REQUEST=str(path))
    run_child([sys.executable,'-u','-m','roshift.speedup.worker',str(path)], env)


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    os.environ['ROSHIFT_SPEED_REQUEST'] = sys.argv[1]
    if request['action']=='summarize':
        from .report import summarize
        print(json.dumps(summarize(Path(request['output'])),indent=2))
        return
    if request['method']=='all' and request['action'] in ('run','validate'):
        for method in ('dense','teal','roshift'):
            child(request, method=method)
        if request['action']=='run':
            from .report import summarize
            print(json.dumps(summarize(Path(request['output'])),indent=2))
        return
    from ..accuracy.integrity import verify_release
    print('[speedup] checking packaged source',flush=True)
    verify_release()
    kernel_only=request['action']=='validate' and request.get('check')=='kernels'
    if not kernel_only:
        verify_checkpoint(request)
    if request['action']=='prepare':
        from .prepare import run
        run(request)
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Select one visible CUDA GPU per invocation')
    torch.set_num_threads(4)
    torch.manual_seed(11)
    torch.backends.cuda.matmul.allow_tf32=False
    torch._inductor.config.compile_threads=1
    sys.path.insert(0,str(ENGINE))
    import support
    common, source, root = support.open_runtime(request['model'], request['artifact_root'])
    common.bench.DATASETS = tuple(request['datasets'])
    if request['action']=='validate':
        from .validation import run
        run(request,common,source)
        return
    import run_method
    sys.argv = [str(ENGINE/'run_method.py'), '--model',request['model'],
                '--method', 'ours' if request['method']=='roshift' else request['method'],
                '--output',request['output'], '--artifact-root',request['artifact_root'],
                '--percents', *map(str,request['percents']), '--repeats',str(request['repeats'])]
    run_method.main()
    dump(Path(request['output'])/request['method']/'request.json',request)


def verify_checkpoint(request):
    from ..data import download_model, verify_model
    path = Path(request['model_path'])
    if not (path/'config.json').exists():
        default = Path(request['data_dir'])/'models'/request['model']
        if path != default:
            raise FileNotFoundError('Explicit model directory does not exist')
        download_model(request['model'],Path(request['data_dir']))
    identity_file=Path(request['data_dir'])/'speedup/identity'/f"{request['model']}.json"
    from ..catalog import asset
    identity=asset('models',request['model']+'.json')
    named_files=[x['path'] for x in identity['weights']]+list(identity['tokenizer_sha256'])+['config.json']
    signatures={}
    for name in named_files:
        source=path/name
        if not source.exists():
            source=path/identity.get('download_file_map',{}).get(name,name)
        stat=source.stat()
        signatures[name]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    signature=dict(directory=str(path.resolve()),files=signatures,expected=identity)
    cached=json.loads(identity_file.read_text()) if identity_file.exists() else None
    if not cached or cached.get('signature')!=signature:
        print('[speedup] hashing original checkpoint',flush=True)
        if not verify_model(request['model'],path)['ok']:
            raise RuntimeError('Checkpoint hash verification failed')
        dump(identity_file,dict(signature=signature,verified=True))
    print('[speedup] checkpoint verified; action='+request['action'],flush=True)


if __name__=='__main__':
    main()
