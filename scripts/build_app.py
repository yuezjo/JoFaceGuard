#!/usr/bin/env python3
"""Build/sign outside iCloud's FileProvider folders, then export a clean ZIP.

Documents may immediately restore FinderInfo onto .app/.mlpackage directories,
making strict codesigning fail even immediately after xattr -c.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / 'build' / 'JoFaceGuard.zip'


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True, cwd=ROOT)


with tempfile.TemporaryDirectory(prefix='JoFaceGuard-build-') as temporary:
    app = Path(temporary) / 'JoFaceGuard.app'
    if '--smoke-only' in sys.argv:
        run('/usr/bin/ditto', '-xk', ARCHIVE, temporary)
    else:
        model = ROOT / 'Resources/Models/SFace.mlpackage'
        if not model.is_dir():
            raise SystemExit('Missing SFace model. Run make model first.')
        binary = app / 'Contents/MacOS/JoFaceGuard'
        resources = app / 'Contents/Resources'
        binary.parent.mkdir(parents=True)
        (resources / 'Models').mkdir(parents=True)
        sources = sorted((ROOT / 'Sources').rglob('*.swift'))
        run('swiftc', '-parse-as-library', '-target', 'arm64-apple-macos14.0',
            '-swift-version', '5', '-O', '-o', binary, *sources)
        for source, destination in [
            (model, resources / 'Models/SFace.mlpackage'),
            (ROOT / 'Resources/Info.plist', app / 'Contents/Info.plist'),
            (ROOT / 'LICENSE', resources / 'LICENSE'),
            (ROOT / 'Resources/SFACE_LICENSE.txt', resources / 'SFACE_LICENSE.txt'),
            (ROOT / 'docs/UPSTREAM.md', resources / 'ATTRIBUTION.md'),
        ]:
            run('/usr/bin/ditto', '--norsrc', source, destination)
        (app / 'Contents/PkgInfo').write_bytes(b'APPL????')
        run('/usr/bin/xattr', '-cr', app)
        run('/usr/bin/codesign', '--force', '--sign', os.environ.get('CODESIGN_ID', '-'),
            '--timestamp=none', app)
        ARCHIVE.parent.mkdir(exist_ok=True)
        ARCHIVE.unlink(missing_ok=True)
        run('/usr/bin/ditto', '-c', '-k', '--norsrc', '--keepParent', app, ARCHIVE)
    run('/usr/bin/codesign', '--verify', '--deep', '--strict', app)
    run(app / 'Contents/MacOS/JoFaceGuard', '--smoke-test')
    print('Verified:', ARCHIVE)
