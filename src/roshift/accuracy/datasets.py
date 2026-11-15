"""Rebuild non-redistributed token inputs from pinned official datasets, on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from functools import lru_cache

from ..catalog import MODELS, asset


def fingerprint(value):
    import torch
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        return {key: fingerprint(item) for key, item in value.items()}
    if isinstance(value, list):
        return [fingerprint(item) for item in value]
    return value


def require_match(value, expected, name):
    if fingerprint(value) != expected:
        raise RuntimeError('Historical token content mismatch; no substitution allowed: '+name)


@lru_cache(maxsize=8)
def load(dataset, config=None, split='train', streaming=True):
    from datasets import load_dataset
    return load_dataset(dataset, config, split=split, streaming=streaming,
                        revision=asset('external_data.json')['revisions'][dataset])


def format_record(row, domain):
    if domain == 0:
        return str(row.get('text', '')).strip()
    if domain == 2:
        question = str(row.get('query', row.get('question', row.get('original_question', '')))).strip()
        answer = str(row.get('response', row.get('answer', row.get('original_response', '')))).strip()
        return (f'Question:\n{question}\n\nAnswer:\n{answer}' if answer else question) if question else ''
    if domain == 4:
        question = str(row.get('text', '')).strip()
        if not question:
            return ''
        parts = [f'Programming problem:\n{question}']
        if row.get('test_list'):
            parts.append('Tests:\n'+'\n'.join(str(x) for x in row['test_list']))
        if str(row.get('code', '')).strip():
            parts.append('Solution:\n'+str(row['code']).strip())
        return '\n\n'.join(parts)
    question = str(row.get('instruction', '')).strip()
    if not question:
        return ''
    parts = [f'Instruction:\n{question}']
    if str(row.get('input', '')).strip():
        parts.append('Input:\n'+str(row['input']).strip())
    answer = str(row.get('output', row.get('response', ''))).strip()
    if answer:
        parts.append('Response:\n'+answer)
    return '\n\n'.join(parts)


def fill(iterator, tokenizer, domain, count):
    values = []
    for row in iterator:
        text = format_record(row, domain)
        if not text:
            continue
        tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        if not tokens:
            continue
        values.extend(tokens[:count-len(values)])
        if len(values) < count and tokenizer.eos_token_id is not None:
            values.append(tokenizer.eos_token_id)
        if len(values) == count:
            return values
    raise RuntimeError('Pinned dataset exhausted before the expected token count')


def tokenizer_for(model, path):
    from transformers import AutoTokenizer
    spec = asset('models', model+'.json')
    for name, expected in spec['tokenizer_sha256'].items():
        source = Path(path)/name
        if not source.exists():
            source = Path(path)/spec.get('download_file_map', {}).get(name, name)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Tokenizer identity differs: '+name)
    return AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)


def input_root(data):
    return Path(data)/'external_data/inputs'


def checked_save(value, output, historical_path=None):
    import torch
    files = asset('external_data.json')['files']
    if historical_path in files:
        require_match(value, files[historical_path]['content'], historical_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        require_match(torch.load(output, weights_only=True, map_location='cpu'), fingerprint(value), str(output))
        return
    temporary = output.with_suffix(output.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(output)


def ensure_roshift(model, path, data):
    import torch
    from .execution import ENGINE
    target = input_root(data)/model
    meta = asset('calibration', model, 'metadata.json')
    marker = target/'data_ready.json'
    identity = dict(model=model, metadata_sha256=hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest(),
                    sources_sha256=hashlib.sha256(json.dumps(asset('external_data.json'), sort_keys=True).encode()).hexdigest())
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['identity'] != identity:
            raise RuntimeError('External-data cache belongs to another source selection')
        for name, expected in record['files'].items():
            if hashlib.sha256((target/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('External-data cache was modified: '+name)
        return target
    target.mkdir(parents=True, exist_ok=True)
    # Copy only frozen metadata/recipes, never the old dataset payloads.
    for source in (ENGINE/'code/inputs'/model).rglob('*.json'):
        destination = target/source.relative_to(ENGINE/'code/inputs'/model)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    tokenizer = tokenizer_for(model, path)
    calibration, selection, ppl, tune, validation, domain_ids = [], [], [], [], [], []
    for i, domain in enumerate(meta['domains']):
        print('[data]', model, domain['name'], flush=True)
        iterator = iter(load(domain['dataset'], domain['config']).shuffle(
            seed=meta['seed']+1009*i, buffer_size=meta['shuffle_buffer']))
        parts = []
        for rows, length in ((domain['blocks'],4096), (domain['blocks'],512),
                             (domain['blocks'],4096), (16,512), (16,512)):
            parts.append(torch.tensor(fill(iterator,tokenizer,i,rows*length),dtype=torch.int64).reshape(rows,length))
        for value, field in zip(parts[:2], ('calibration_sha256','selection_sha256')):
            if fingerprint(value)['sha256'] != domain[field]:
                raise RuntimeError('Historical domain replay mismatch: '+domain['name']+' '+field)
        calibration.append(parts[0]); selection.append(parts[1]); ppl.append(parts[2])
        tune.append(parts[3]); validation.append(parts[4]); domain_ids.extend([i]*domain['blocks'])
    prefix = f'accuracy/_engine/code/inputs/{model}/'
    for name, parts in (('input_ids.pt',calibration), ('selection_input_ids.pt',selection)):
        relative = 'root/mixed_calibration/'+name
        checked_save(torch.cat(parts), target/relative, prefix+relative)
    for offset, split in enumerate(('train','test')):
        iterator = iter(load('google-research-datasets/mbpp','full',split).shuffle(
            seed=meta['seed']+1009*4+offset,buffer_size=meta['shuffle_buffer']))
        value = torch.tensor(fill(iterator,tokenizer,4,16*512),dtype=torch.int64).reshape(16,512)
        (tune if offset==0 else validation).append(value)
    names = [row['name'] for row in meta['domains']]
    heldout = dict(ppl_input_ids=torch.cat(ppl), ppl_sequence_domain_ids=torch.tensor(domain_ids,dtype=torch.int16),
                   ppl_domain_names=names, lmhead_tune_input_ids=torch.cat(tune),
                   lmhead_validation_input_ids=torch.cat(validation),
                   lmhead_sequence_domain_ids=torch.arange(5,dtype=torch.int16).repeat_interleave(16),
                   lmhead_domain_names=names+['code_mbpp'])
    checked_save(heldout,target/'heldout/heldout_tokens.pt',prefix+'heldout/heldout_tokens.pt')
    rebuild_wikitext(model, tokenizer, target)
    files = {p.relative_to(target).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in target.rglob('*') if p.is_file() and p!=marker}
    marker.write_text(json.dumps(dict(identity=identity,files=files,
        historical_heldout_reference=(prefix+'heldout/heldout_tokens.pt') in asset('external_data.json')['files']),indent=2)+'\n')
    return target


def rebuild_wikitext(model, tokenizer, target):
    import numpy as np
    metadata = target/'evalset/metadata.json'
    meta = json.loads(metadata.read_text()) if metadata.exists() else None
    rows = list(load('Salesforce/wikitext','wikitext-103-raw-v1','test'))
    indices = list(range(len(rows)))
    random.Random(42).shuffle(indices)
    selected, arrays = [], []
    # Older Llama samples index nonempty rows and omit EOS. Later samples
    # index raw rows and append EOS. Preserve the recorded protocols.
    legacy_llama = model in ('llama31','llama2')
    if meta and legacy_llama:
        rows = [row for row in rows if str(row['text']).strip()]
    selections = meta['sampled_row_indices'] if meta else [range(i,len(rows)) for i in indices]
    for recorded in selections:
        values, used = [], []
        for index in recorded:
            text = rows[index]['text']
            if not str(text).strip():
                continue
            tokens = tokenizer(str(text),add_special_tokens=False)['input_ids']
            if not tokens:
                continue
            values.extend(tokens)
            if not legacy_llama and tokenizer.eos_token_id is not None:
                values.append(tokenizer.eos_token_id)
            used.append(index)
            if len(values)>=4096:
                break
        if len(values)>=4096:
            arrays.append(values[:4096]); selected.append(used)
        if len(arrays)==4:
            break
    if len(arrays)!=4 or (meta and selected!=meta['sampled_row_indices']):
        raise RuntimeError('WikiText row selection differs')
    output = target/'evalset/input_ids.bin'
    content = np.asarray(arrays,dtype=np.int32).tobytes()
    reference = asset('external_data.json')['files'].get(f'accuracy/_engine/code/inputs/{model}/evalset/input_ids.bin')
    if reference and hashlib.sha256(content).hexdigest()!=reference['file_sha256']:
        raise RuntimeError('Historical WikiText token mismatch')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(content)
    if not meta:
        meta = dict(dataset_name='Salesforce/wikitext',dataset_config='wikitext-103-raw-v1',split='test',
                    num_samples=4,seq_len=4096,dtype='int32',data_file='input_ids.bin',seed=42,
                    sampled_row_indices=selected, historical_reference_available=False)
        metadata.write_text(json.dumps(meta,indent=2)+'\n')


def ensure_larosa(model, path, data):
    import torch
    target = Path(data)/'external_data/larosa'/model
    metadata = asset('larosa',model,'calibration_metadata.json')
    output = target/'calibration_input_ids.pt'
    reference = f'assets/larosa/{model}/calibration_input_ids.pt'
    if output.exists():
        require_match(torch.load(output,map_location='cpu',weights_only=True),
                      asset('external_data.json')['files'][reference]['content'],reference)
    else:
        tokenizer = tokenizer_for(model,path)
        text = ''.join(row['text']+'\n\n' for row in load('Salesforce/wikitext','wikitext-2-raw-v1'))
        if hashlib.sha256(text.encode()).hexdigest()!=metadata['text_sha256']:
            raise RuntimeError('LaRoSA WikiText content changed')
        stream = tokenizer(text,add_special_tokens=True,truncation=False,return_tensors='pt').input_ids[0]
        ids = torch.stack([stream[start:start+2048] for start in metadata['sampling']['starts']])
        checked_save(ids,output,reference)
    metadata['historical_serialized_token_sha256'] = metadata['token_sha256']
    metadata['token_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    (target/'calibration_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=MODELS)
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--tokenizer-path',type=Path)
    parser.add_argument('--method',choices=('roshift','larosa','both'),default='both')
    args = parser.parse_args()
    from .execution import work_environment
    root, env = work_environment(args.data_dir)
    os.environ.update(env)
    sys.path[:0] = env['PYTHONPATH'].split(os.pathsep)
    if args.tokenizer_path:
        path = args.tokenizer_path
    else:
        from huggingface_hub import snapshot_download
        spec = asset('models',args.model+'.json')
        path = snapshot_download(spec['hf_id'],revision=spec['download_revision'],
            allow_patterns=[spec.get('download_file_map',{}).get(name,name)
                            for name in spec['tokenizer_sha256']],max_workers=2)
    for method in (('roshift','larosa') if args.method=='both' else (args.method,)):
        (ensure_roshift if method=='roshift' else ensure_larosa)(args.model,path,root)
    print('CPU data reproduction verified; no model weights or GPU were loaded.')


if __name__ == '__main__':
    main()
