"""Check checkpoint availability and identities without downloading weights."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args()
    from roshift.accuracy.execution import work_environment
    root,env=work_environment(args.data_dir)
    os.environ.update(env)
    sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
    from huggingface_hub import HfApi,hf_hub_download
    from roshift.catalog import model_config,asset
    from roshift.data import config_matches
    from roshift.io import digest,atomic_json
    rows=[]
    for name in ('llama31','llama2','mistral','qwen25','qwen3'):
        config=model_config(name)
        revision=config['download_revision'] or 'main'
        info=HfApi().model_info(config['hf_id'],revision=revision,files_metadata=True)
        available={f.rfilename:f for f in info.siblings}
        weights=[]
        for weight in config['weights']:
            f=available.get(weight['path'])
            actual=getattr(getattr(f,'lfs',None),'sha256',None)
            weights.append(dict(file=weight['path'],passed=actual==weight['sha256']))
        tokens=[]
        for file,sha in config['tokenizer_sha256'].items():
            remote=config.get('download_file_map',{}).get(file,file)
            if remote not in available:
                tokens.append(dict(file=file,passed=False,reason='missing_remote_file'))
                continue
            path=Path(hf_hub_download(config['hf_id'],remote,revision=info.sha))
            tokens.append(dict(file=file,remote=remote,passed=digest(path)==sha))
        native_path=Path(hf_hub_download(config['hf_id'],'config.json',revision=info.sha))
        semantic_ok=config_matches(json.loads(native_path.read_text()),asset('models',name+'_native_config.json'))
        row=dict(model=name,requested_revision=revision,resolved_revision=info.sha,config_matches=semantic_ok,
                 weights=weights,tokenizer=tokens,
                 passed=semantic_ok and all(r['passed'] for r in [*weights,*tokens]))
        rows.append(row)
        atomic_json(root/'download_contracts.json',dict(passed=all(r['passed'] for r in rows),
                    completed_models=len(rows),checks=rows,weights_downloaded=False))
        print(name,'passed' if row['passed'] else 'FAILED',flush=True)
    if not all(r['passed'] for r in rows):
        raise RuntimeError('Remote checkpoint identity differs from frozen hashes')


if __name__=='__main__':
    main()
