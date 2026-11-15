"""Prepared-checkpoint scope audit and explicitly partial task smoke tests."""
import gc
import sys


def main():
    from roshift.accuracy.execution import parser,make_request
    p=parser()
    args=p.parse_args(['evaluate',*sys.argv[1:]])
    if not args.execute or args.limit is None:
        p.error('Checkpoint smoke validation requires --execute and an explicit --limit')
    request=make_request(args)
    from roshift.accuracy.worker import setup
    root,path,work=setup(request)
    from roshift.accuracy.audit import run as audit
    from roshift.accuracy.evaluate import run as evaluate
    audit(request,path,work,root)
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    evaluate(request,path,work,root)


if __name__=='__main__':
    main()
