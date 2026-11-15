"""Route the optional Orin command without changing the original CLI."""
import sys


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == 'speedup-orin':
        from .speedup_orin.execution import main as run
        return run(values[1:])
    from .cli import main as original
    return original(values)


if __name__ == '__main__':
    main()
