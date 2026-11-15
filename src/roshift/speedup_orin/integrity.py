"""Independent integrity receipt for the Orin backend and shared inputs."""
import json
from pathlib import Path
from ..io import digest


def source_identity():
    return digest(Path(__file__).with_name('release_sources.json'))


def verify_release():
    package = Path(__file__).resolve().parents[1]
    manifest = Path(__file__).with_name('release_sources.json')
    for relative, expected in json.loads(manifest.read_text()).items():
        path = (package/relative).resolve()
        if not path.is_relative_to(package) or not path.is_file() or digest(path) != expected:
            raise RuntimeError('Orin release file differs: '+relative)
    return source_identity()
