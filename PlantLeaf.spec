# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [
    ('themes', 'themes'),
    ('assets', 'assets'),
    ('README.txt', '.'),
    ('licenses.txt', '.')   
]
binaries = []
hiddenimports = []
tmp_ret = collect_all('pyqtgraph')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=binaries,
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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PlantLeaf',
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
    icon=['assets/logo_for_app.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PlantLeaf',
)
app = BUNDLE(
    coll,
    name='PlantLeaf.app',
    icon='assets/logo_for_app.icns',
    bundle_identifier='com.tommydev.plantleaf',
    info_plist={
        'CFBundleName': 'PlantLeaf',
        'CFBundleDisplayName': 'PlantLeaf',
        'CFBundleIdentifier': 'com.tommydev.plantleaf',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'PlantLeaf Voltage File',
                'CFBundleTypeRole': 'Editor',
                'CFBundleTypeIconFile': 'logo_for_app.icns',
                'LSHandlerRank': 'Owner',
                'LSItemContentTypes': ['com.tommydev.plantleaf.pvolt'],
            },
            {
                'CFBundleTypeName': 'PlantLeaf Audio File',
                'CFBundleTypeRole': 'Editor',
                'CFBundleTypeIconFile': 'logo_foor_app.icns',
                'LSHandlerRank': 'Owner',
                'LSItemContentTypes': ['com.tommydev.plantleaf.paudio'],
            }
        ],
        'UTExportedTypeDeclarations': [
            {
                'UTTypeIdentifier': 'com.tommydev.plantleaf.pvolt',
                'UTTypeConformsTo': ['public.data'],
                'UTTypeDescription': 'PlantLeaf Voltage Data',
                'UTTypeTagSpecification': {
                    'public.filename-extension': ['pvolt'],
                },
            },
            {
                'UTTypeIdentifier': 'com.tommydev.plantleaf.paudio',
                'UTTypeConformsTo': ['public.data'],
                'UTTypeDescription': 'PlantLeaf Audio Data',
                'UTTypeTagSpecification': {
                    'public.filename-extension': ['paudio'],
                },
            }
        ],
    },
)