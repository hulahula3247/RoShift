"""Check the installed package's numerical and supplemental source hashes."""
import json
import sys

from roshift.accuracy.execution import ENGINE
from roshift.accuracy.integrity import verify_release


def main():
    release_hash = verify_release()
    sys.path.insert(0, str(ENGINE / 'code/tools'))
    from verify_package import verify
    verify(ENGINE / 'code')
    print(json.dumps(dict(passed=True, supplemental_manifest_sha256=release_hash,
                         gpu_initialized=False), indent=2))


if __name__ == '__main__':
    main()
