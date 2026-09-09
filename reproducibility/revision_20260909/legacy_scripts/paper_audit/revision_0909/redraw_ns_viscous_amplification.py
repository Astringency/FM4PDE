"""Reproduce the analytic illustration without importing or running inference."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

PAPER = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent / 'ns_viscous_figure'
CODE = Path('/home/tat512/C01Python/FM4PDE')
EXPORTER = CODE / 'plot/export_ns_true_trajectory.py'
sys.path.insert(0, str(EXPORTER.parent))
from export_ns_true_trajectory import viscous_amplification_figure, plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analytic_nodes(source):
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    return [n for n in main.body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in {'radius', 'z', 'exact', 'secant', 'threshold'}]


AUDIT.mkdir(parents=True, exist_ok=True)
original = subprocess.run(['git', 'show', 'HEAD:plot/export_ns_true_trajectory.py'],
                          cwd=CODE, capture_output=True, text=True, check=True).stdout
old_nodes, current_nodes = analytic_nodes(original), analytic_nodes(EXPORTER.read_text())
assert len(old_nodes) == len(current_nodes) == 5
assert [ast.dump(n) for n in old_nodes] == [ast.dump(n) for n in current_nodes]
values = dict(np=np, nu=.001, T=1.)
exec(compile(ast.Module(body=old_nodes, type_ignores=[]), '<original analytic expressions>', 'exec'), values)
radius, exact, secant, threshold = [values[k] for k in ['radius', 'exact', 'secant', 'threshold']]
assert len(radius) == 1601 and radius[0] == 0 and radius[-1] == 64
original_report = json.loads((PAPER / 'source_data/ns_temporal_diagnostic_report.json').read_text())
assert threshold == original_report['sign_change_radius']
for point in original_report['amplification_points']:
    i = int(np.flatnonzero(radius == point['radius'])[0])
    assert exact[i] == point['exact_viscous_amplification']
    assert secant[i] == point['endpoint_secant_amplification']
fig, font = viscous_amplification_figure(radius, exact, secant, threshold)
ax = fig.axes[0]
assert np.array_equal(ax.lines[0].get_xdata(), radius)
assert np.array_equal(ax.lines[0].get_ydata(), exact)
assert np.array_equal(ax.lines[1].get_ydata(), secant)
assert ax.get_xlim() == (0., 64.) and ax.get_ylim() == (-1.08, 1.05)
assert [ax.lines[i].get_color() for i in (0, 1)] == ['#256493', '#aa4b32']
assert [ax.lines[i].get_linestyle() for i in (0, 1)] == ['-', '--']
assert np.array_equal(ax.lines[3].get_xdata(), [threshold, threshold])
assert r'\mathcal{T}=1' in ax.get_title()
for ext in ['pdf', 'png']:
    fig.savefig(AUDIT / f'ns_viscous_amplification.{ext}', dpi=190, bbox_inches='tight')
plt.close(fig)
xml = subprocess.run(['pdftotext', '-bbox', str(AUDIT / 'ns_viscous_amplification.pdf'), '-'],
                     capture_output=True, text=True, check=True).stdout
page = next(n for n in ET.fromstring(xml).iter() if n.tag.endswith('page'))
width, height = [float(page.attrib[k]) for k in ['width', 'height']]
for word in (n for n in page.iter() if n.tag.endswith('word')):
    assert float(word.attrib['xMin']) >= -.5 and float(word.attrib['yMin']) >= -.5
    assert float(word.attrib['xMax']) <= width + .5 and float(word.attrib['yMax']) <= height + .5
final_font = 10 * (5.4 * 72) / width
assert final_font >= 9
preview = AUDIT / 'preview.tex'
preview.write_text(r'''\documentclass{article}
\usepackage{graphicx}
\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=6in]{geometry}
\pagestyle{empty}
\begin{document}
\begin{figure}[!htbp]
\centering\includegraphics[width=0.9\linewidth]{ns_viscous_amplification.pdf}
\caption{Linear viscous amplification at $\nu=10^{-3}$ and $\mathcal T=1$.}
\end{figure}
\end{document}
''')
subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'preview.tex'],
               cwd=AUDIT, stdout=subprocess.DEVNULL, check=True)
log = (AUDIT / 'preview.log').read_text()
assert 'Overfull' not in log
subprocess.run(['pdftoppm', '-png', '-singlefile', '-r', '150', 'preview.pdf', 'preview'],
               cwd=AUDIT, stdout=subprocess.DEVNULL, check=True)
report = dict(
    status='rendered; awaiting visual inspection',
    chart_contract=dict(question='Compare exact linear viscous decay with the endpoint-secant amplification.',
                        takeaway='The endpoint-secant factor changes sign at the retained analytic threshold.',
                        family='two continuous analytic lines', grid_points=1601, radius_range=[0, 64],
                        surface='static supplement figure at 0.9 times a six-inch text width',
                        palette={'exact': '#256493, solid', 'endpoint_secant': '#aa4b32, dashed'},
                        interpretation=original_report['interpretation']),
    no_inference_executed=True, analytic_expression_ast_unchanged=True,
    all_1601_plot_coordinates_verified=True, six_archived_anchor_points_exactly_preserved=True,
    sign_change_radius=threshold, xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
    curve_sha256=hashlib.sha256(np.stack([radius, exact, secant]).tobytes()).hexdigest(),
    original_report_sha256=sha(PAPER / 'source_data/ns_temporal_diagnostic_report.json'),
    exporter_sha256=sha(EXPORTER), redraw_script_sha256=sha(Path(__file__)), font_path=font,
    nominal_ordinary_font_pt=10, pdf_width_bp=width, pdf_height_bp=height,
    final_width_in=5.4, final_ordinary_font_bp=final_font,
    all_pdf_text_within_bounds=True, tex_preview_no_overfull=True,
    outputs={f'ns_viscous_amplification.{ext}': sha(AUDIT / f'ns_viscous_amplification.{ext}') for ext in ['pdf', 'png']})
(AUDIT / 'qa.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
