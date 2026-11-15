"""Prepare pinned LaRoSA WikiText2 activation PCA on one GPU; dry-run by default."""
import argparse
import gc
import json
from pathlib import Path

from common import CORE, dump, official_function, provenance, sha
from calibration import PAPER_PROTOCOL,SOURCE_PROTOCOL,REVISION,prepare_tokens
from native import validate_native_model


def capture_first_layer_inputs(model,input_ids):
    """Let the native model compute embedding scaling, masks and RoPE once."""
    if len(set(getattr(model.config,'layer_types',[]) or []))>1:
        raise NotImplementedError('Layerwise PCA replay requires homogeneous native mask type; mixed sliding/full layers need a separate replay port')
    captured={}
    class CaptureDone(Exception):
        pass
    def first(module,positional,keyword):
        captured['hidden']=positional[0] if positional else keyword['hidden_states']
        captured['kwargs']={k:v for k,v in keyword.items() if k!='hidden_states'}
        raise CaptureDone()
    handle=model.model.layers[0].register_forward_pre_hook(first,with_kwargs=True)
    try:
        try:
            model.model(input_ids=input_ids,use_cache=False)
        except CaptureDone:
            pass
    finally:
        handle.remove()
    return captured['hidden'],captured['kwargs']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--calibration-protocol',choices=(PAPER_PROTOCOL,SOURCE_PROTOCOL),default=PAPER_PROTOCOL)
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--dataset-revision',default=REVISION)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(dict(model=str(args.model_path), output=str(args.output),
        calibration_protocol=args.calibration_protocol,seed=args.seed,dataset_revision=args.dataset_revision,
        precision='BF16 dense calibration; official FP32 accumulator + FP64 eigh',
        gpu='one visible GPU, no TP/PP', execute=args.execute), indent=2), flush=True)
    if not args.execute:
        return
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(model_config_sha256=sha(args.model_path/'config.json'), source=provenance(),
                    calibration_protocol=args.calibration_protocol,seed=args.seed,dataset_revision=args.dataset_revision)
    marker = args.output/'preparation_identity.json'
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise RuntimeError('Changed preparation identity; use another output directory')
    dump(marker, identity)
    token_path = args.output/'calibration_input_ids.pt'
    ids,calibration_meta=prepare_tokens(args.model_path,args.output,args.calibration_protocol,args.seed,args.dataset_revision)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,
        device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)
    native_architecture=validate_native_model(model)
    eigen = official_function('gen_act/grab_acts_rotate_diff_Q_llama.py', 'perform_eigen_decomp')
    with torch.inference_mode():
        hidden,kwargs=capture_first_layer_inputs(model,ids.to('cuda:0'))
        for index, layer in enumerate(model.model.layers):
            activations = {}
            def capture(name):
                def hook(module, call, keyword):
                    activations[name] = call[0] if call else keyword['hidden_states']
                return hook
            handles = [layer.self_attn.register_forward_pre_hook(capture('self_attn'), with_kwargs=True),
                       layer.mlp.register_forward_pre_hook(capture('mlp'), with_kwargs=True)]
            try:
                result = layer(hidden, **kwargs)
                hidden = result[0] if isinstance(result, tuple) else result
            finally:
                for handle in handles:
                    handle.remove()
            for group, x in activations.items():
                directory = args.output/f'histograms/layer-{index}/{group}'
                completed = directory/'pca.json'
                if completed.exists():
                    prior = json.loads(completed.read_text())
                    if prior.get('status')=='complete' and prior.get('finite') is True and prior['D_sha256'] == sha(directory/'D.pt') and prior['inv_D_sha256'] == sha(directory/'inv_D.pt'):
                        continue
                    raise RuntimeError(f'Changed PCA artifacts: {directory}')
                # Upstream accumulates this FP64 batch sum into H initialized
                # as FP32; preserve that cast before its damping/eigh helper.
                x64 = x.double()
                covariance = torch.sum(x64.mT @ x64, dim=0).float()
                del x64
                eigenvalues, q = eigen(covariance)
                if not torch.isfinite(q).all():
                    raise RuntimeError('Nonfinite PCA basis')
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(q.cpu(), directory/'D.pt')
                torch.save(q.T.contiguous().cpu(), directory/'inv_D.pt')
                dump(completed, dict(status='complete', D_sha256=sha(directory/'D.pt'),
                    inv_D_sha256=sha(directory/'inv_D.pt'), input_shape=list(x.shape),
                    finite=bool(torch.isfinite(q).all()), dtype=str(q.dtype)))
                del covariance, eigenvalues, q
            dump(args.output/'q_progress.json', dict(status='running', last_layer=index,
                layers_total=len(model.model.layers), token_sha256=sha(token_path)))
            print(f'[LaRoSA PCA] layer {index+1}/{len(model.model.layers)}', flush=True)
            del activations
            gc.collect()
            torch.cuda.empty_cache()
    dump(args.output/'q_manifest.json', dict(status='complete', layers=len(model.model.layers),
        token_sha256=sha(token_path), source=provenance(), inference_audit_required=True,
        native_architecture=native_architecture,
        calibration_protocol=args.calibration_protocol,calibration_metadata_sha256=sha(args.output/'calibration_metadata.json')))


if __name__ == '__main__':
    main()
