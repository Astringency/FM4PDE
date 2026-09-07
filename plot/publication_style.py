"""Paper figure typography with an explicit, verifiable Times New Roman font."""
from pathlib import Path


def use_times_new_roman():
    import matplotlib as mpl
    from matplotlib import font_manager
    for name in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf']:
        path = Path('/mnt/c/Windows/Fonts') / name
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    font = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
    mpl.rcParams.update({
        'font.family': 'Times New Roman', 'font.serif': ['Times New Roman'],
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold', 'mathtext.sf': 'Times New Roman',
        'mathtext.tt': 'Times New Roman', 'mathtext.cal': 'Times New Roman:italic',
        'mathtext.fallback': 'stix',
    })
    return font
