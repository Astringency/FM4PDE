"""Paper figure typography with an explicit, verifiable Times New Roman font."""
from pathlib import Path
import math


def error_number(value, *, signed=False):
    """Format a percentage without rounding a nonzero error to zero."""
    if not math.isfinite(value):
        raise ValueError('A displayed error must be finite')
    sign = '+' if signed else ''
    if 0 < abs(value) < .01:
        if abs(value) >= .0001:
            return format(value, sign + '.2g')
        mantissa, exponent = format(value, sign + '.1e').split('e')
        return f'{mantissa}\\times10^{{{int(exponent)}}}'
    return format(value, sign + '.2f')


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
