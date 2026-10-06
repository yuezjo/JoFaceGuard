#!/usr/bin/env python3
"""Package a supplied square PNG as native icons without changing its artwork.

Personal artwork stays outside Git; its rights are separate from the app's MIT code.
Usage: python3 scripts/make_personal_icon.py /absolute/path/to/icon.png
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'Resources' / 'PersonalIcon'

if len(sys.argv) != 2:
    raise SystemExit('Pass the source PNG path.')
source = Path(sys.argv[1]).expanduser().resolve(strict=True)
OUT.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, OUT / 'BrandIcon.png')

with tempfile.TemporaryDirectory(prefix='JoFaceGuard-icon-') as temporary:
    iconset = Path(temporary) / 'AppIcon.iconset'
    iconset.mkdir()
    for points in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            pixels = points * scale
            suffix = '' if scale == 1 else '@2x'
            target = iconset / f'icon_{points}x{points}{suffix}.png'
            subprocess.run(['/usr/bin/sips', '-z', str(pixels), str(pixels), str(source), '--out', str(target)],
                           check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['/usr/bin/iconutil', '-c', 'icns', str(iconset), '-o', str(OUT / 'AppIcon.icns')], check=True)
    for scale in (1, 2):
        target = OUT / ('MenuIcon.png' if scale == 1 else 'MenuIcon@2x.png')
        subprocess.run(['/usr/bin/sips', '-z', str(22 * scale), str(22 * scale), str(source), '--out', str(target)],
                       check=True, stdout=subprocess.DEVNULL)
print('Prepared local artwork:', OUT)
