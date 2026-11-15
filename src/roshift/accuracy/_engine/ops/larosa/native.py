"""Native family contract: retain model attention, norms, embeddings and biases."""
SUPPORTED=('llama','mistral','qwen2','qwen3','gemma')


def validate_native_model(model):
    if model.config.model_type not in SUPPORTED:
        raise NotImplementedError(f'Unaudited LaRoSA native family: {model.config.model_type}')
    if getattr(model.config,'pretraining_tp',1)!=1:
        raise ValueError('Only single-device, non-TP inference is supported')
    hidden=model.config.hidden_size
    intermediate=model.config.intermediate_size
    rows=[]
    for index,layer in enumerate(model.model.layers):
        projections={name:getattr(layer.self_attn if name in ('q','k','v','o') else layer.mlp,name+'_proj')
                     for name in ('q','k','v','o','gate','up','down')}
        for name in ('q','k','v','gate','up'):
            if projections[name].in_features!=hidden:
                raise ValueError(f'Unexpected {name} input width in layer {index}')
        if projections['down'].in_features!=intermediate:
            raise ValueError('Unexpected down projection input width')
        if projections['o'].in_features!=projections['q'].out_features:
            raise ValueError('Attention O width differs from query output width')
        rows.append(dict(layer=index,input_norm=type(layer.input_layernorm).__name__,
            post_attention_norm=type(layer.post_attention_layernorm).__name__,
            q_norm=type(layer.self_attn.q_norm).__name__ if hasattr(layer.self_attn,'q_norm') else None,
            k_norm=type(layer.self_attn.k_norm).__name__ if hasattr(layer.self_attn,'k_norm') else None,
            activation=type(layer.mlp.act_fn).__name__,
            projections={name:dict(input_width=p.in_features,output_width=p.out_features,bias=p.bias is not None)
                         for name,p in projections.items()}))
    return dict(model_type=model.config.model_type,embedding=type(model.get_input_embeddings()).__name__,
        layers=rows,native_forward_preserved=True,full_pretrained_GPU_validation=False)
