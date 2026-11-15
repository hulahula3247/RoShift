"""Reconstruct historical timing prompts; distribute hashes, not dataset text."""
import hashlib
import json
from pathlib import Path

from .runtime import ROOT, DATASETS, dump
from ..catalog import asset


def text_rows(name, rows):
    for row in rows:
        if name in ('wikitext2','alpaca'):
            text = row.get('text','')
            if not text.strip():
                continue
        elif name=='gsm8k':
            if not row.get('question','').strip():
                continue
            text = f"Question: {row['question'].strip()}\nAnswer: {row.get('answer','').strip()}"
        else:
            if not row.get('question_stem','').strip():
                continue
            choices = ' '.join(f'{label}. {value}' for label,value in zip(row['choices']['label'],row['choices']['text']))
            text = f"Question: {row['question_stem'].strip()}\nChoices: {choices}\nAnswer: {row.get('answerKey','')}"
        yield text


def prepare(request):
    import torch
    from tokenizers import Tokenizer
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    metadata = json.loads((ROOT/'assets/prompts.json').read_text())
    output = Path(request['artifact_root'])/'prompts'
    expected = metadata['models'][request['model']]

    def verify(values):
        for name,item in expected.items():
            ids=values[name].to(dtype=torch.int64, device='cpu').contiguous()
            if len(ids)!=item['tokens'] or hashlib.sha256(ids.numpy().tobytes()).hexdigest()!=item['sha256']:
                raise RuntimeError('Timing prompt content mismatch: '+name)

    if (output/'prompts.pt').exists():
        verify(torch.load(output/'prompts.pt',map_location='cpu',weights_only=True))
        return
    identity = asset('models','llama31.json')
    base_path = Path(request['data_dir'])/'models/llama31/tokenizer.json'
    if request['model']=='llama31':
        base_path = Path(request['model_path'])/'tokenizer.json'
    if not base_path.exists():
        base_path = Path(hf_hub_download(identity['hf_id'], 'tokenizer.json', revision=identity['download_revision']))
    if hashlib.sha256(base_path.read_bytes()).hexdigest()!=identity['tokenizer_sha256']['tokenizer.json']:
        raise RuntimeError('Llama timing-source tokenizer hash differs')
    original=Tokenizer.from_file(str(base_path))
    target=Tokenizer.from_file(str(Path(request['model_path'])/'tokenizer.json'))
    sources={'wikitext2':('Salesforce/wikitext','wikitext-2-raw-v1'),
             'alpaca':('tatsu-lab/alpaca',None),'gsm8k':('openai/gsm8k','main'),
             'obqa':('allenai/openbookqa','main')}
    values={}
    for name in DATASETS:
        repo,config=sources[name]
        dataset=load_dataset(repo,config,split='train',streaming=True,revision=metadata['revisions'][repo])
        texts=[]
        for text in text_rows(name,dataset):
            texts.append(text)
            if len(texts)==2048:
                break
        tokens=original.encode('\n\n'.join(texts),add_special_tokens=False).ids[:2048]
        if len(tokens)!=2048:
            raise RuntimeError('Insufficient timing-source tokens')
        tokens[0]=128000
        text=original.decode(tokens,skip_special_tokens=True)
        values[name]=torch.tensor(target.encode(text,add_special_tokens=False).ids[:2048],dtype=torch.int64)
    verify(values)
    output.mkdir(parents=True,exist_ok=True)
    torch.save(values,output/'prompts.pt')
    dump(output/'provenance.json',dict(protocol='historical row0; Llama decode then target retokenization',
         revisions=metadata['revisions'],verified_content=expected,shuffle=False))
