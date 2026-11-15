"""Versioned WikiText train calibration; exact tokens and source identity persist."""
import hashlib
import json
import random
from pathlib import Path

from common import dump, sha

DATASET='Salesforce/wikitext'
REVISION='b08601e04326c79dfdd32d625aee71d232d685c3'
PAPER_PROTOCOL='paper_random16x2048_train_v1'
SOURCE_PROTOCOL='released_first10x2048_train_reference_v1'


def tokenizer_identity(model_path):
    root=Path(model_path)
    names=('tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt')
    return {name:sha(root/name) for name in names if (root/name).exists()}


def prepare_tokens(model_path,output,protocol=PAPER_PROTOCOL,seed=1234,revision=REVISION):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    output=Path(output)
    token_path=output/'calibration_input_ids.pt'
    metadata_path=output/'calibration_metadata.json'
    expected=dict(protocol=protocol,dataset=DATASET,revision=revision,subset='wikitext-2-raw-v1',
                  split='train',seed=seed,tokenizer_sha256=tokenizer_identity(model_path))
    if token_path.exists() or metadata_path.exists():
        if not token_path.exists() or not metadata_path.exists():
            raise RuntimeError('Incomplete prior calibration files; inspect before restarting')
        metadata=json.loads(metadata_path.read_text())
        if metadata['identity']!=expected or metadata['token_sha256']!=sha(token_path):
            raise RuntimeError('Calibration fingerprint changed; do not reuse a different PCA protocol')
        return torch.load(token_path,map_location='cpu',weights_only=True),metadata
    tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id=tokenizer.eos_token_id
    wiki=load_dataset(DATASET,'wikitext-2-raw-v1',split='train',streaming=True,revision=revision)
    if protocol==PAPER_PROTOCOL:
        text=''.join(row['text']+'\n\n' for row in wiki)
        stream=tokenizer(text,add_special_tokens=True,truncation=False,return_tensors='pt').input_ids[0]
        if len(stream)<2048+16:
            raise ValueError('WikiText training token stream unexpectedly short')
        starts=random.Random(seed).sample(range(len(stream)-2048+1),16)
        ids=torch.stack([stream[start:start+2048] for start in starts])
        sampling=dict(starts=starts,special_tokens='one stream with tokenizer add_special_tokens=True',
            algorithm='Python random.Random(seed).sample distinct start offsets; windows may overlap',
            seed_origin='recipient-declared; paper and public source do not provide a random search/PCA seed',
            training_tokens=len(stream),sequence_count=16,sequence_length=2048)
    elif protocol==SOURCE_PROTOCOL:
        text=''.join(row['text']+'\n\n' for row in wiki.take(500))
        ids=tokenizer(text,truncation=True,return_tensors='pt',max_length=2048,
            return_overflowing_tokens=True,padding='max_length').input_ids[:10]
        sampling=dict(algorithm='pinned public script first500 rows, first10 tokenizer overflow blocks',
                      sequence_count=10,sequence_length=2048,seed_used=False)
    else:
        raise ValueError(f'Unknown calibration protocol: {protocol}')
    if tuple(ids.shape)!=(sampling['sequence_count'],2048) or ids.dtype!=torch.int64:
        raise ValueError('Unexpected calibration token shape/dtype')
    output.mkdir(parents=True,exist_ok=True)
    temporary=token_path.with_suffix('.tmp')
    torch.save(ids,temporary)
    temporary.replace(token_path)
    metadata=dict(identity=expected,sampling=sampling,token_sha256=sha(token_path),
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),selection_objective='full causal TRAIN calibration PPL',
        test_or_benchmark_data_used=False)
    dump(metadata_path,metadata)
    return ids,metadata
