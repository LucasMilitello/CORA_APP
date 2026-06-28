# -*- mode: python ; coding: utf-8 -*-
# PT: Spec do PyInstaller para empacotar a GUI CORA em modo janela. | EN: PyInstaller spec for packaging the CORA GUI in windowed mode.

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

# PT: Coleta dados e imports dinamicos necessarios para execucao no executavel. | EN: Collects data and dynamic imports required by the executable.
datas = []
hiddenimports = ['threadpoolctl', 'matplotlib.backends.backend_tkagg']
datas += collect_data_files('matplotlib')
hiddenimports += collect_submodules('sklearn')


a = Analysis(
    ['cora_projeto/run_cora_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# PT: EXE sem console, apropriado para aplicacao desktop tkinter. | EN: Console-free EXE suitable for the Tkinter desktop application.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CORA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# PT: Coleta final em pasta dist/CORA. | EN: Final collection in the dist/CORA folder.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CORA',
)
