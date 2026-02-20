# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['BleDataProcessor.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('tidal/models', 'tidal/models'),
        ('forced/models', 'forced/models'),
		('models', 'models'),		
		('UrineTest', 'UrineTest')
    ],
    hiddenimports=[
		'forced.GLI_2012_referencevalues',
	],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'forced.forced_calculations',
		'matplotlib',           # ← Full exclusion
        'matplotlib.pyplot',     # ← Specific submodule
        'matplotlib.animation',  # ← FuncAnimation
        'tkinter',               # ← GUI backend
        'PIL'                   # ← Image deps (if any)
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BleDataProcessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir="C:/ReMeDi_TMP",
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
