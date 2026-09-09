# Ensemble and spatial-spectrum figures

Audience: JMLR scientific readers. Purpose: show how averaging three conditional predictions changes each physical field, and separate predicted spectral power from spectral error.

Sources: the registered ensemble analysis, summary.csv and spectral_shells.npz. Independent unit: 32 physical inputs per PDE. Main comparison: mean of three fields minus predeclared seed 0; 21 separate field comparisons across 11 PDEs. Development exports explicitly retain their partial-study status outside the article.

Deliverable: standalone PDF and PNG, 6.2-inch width, Times New Roman and consistent serif mathematics. No decorative badges.

1. Paired changes: horizontal point and interval plot, one row per PDE/field. Points are mean percentage-point changes, intervals are the registered Bonferroni-adjusted paired bootstrap intervals. Zero is the reference; field a is blue and field u is gold. Both fields remain separate.
2. Each PDE: energy spectra and error spectra in two columns, with separate rows for a and u (Burgers u only). Plot mean per-input normalized shell energies; show truth, seed 0 and mean of 3. Normalize by full reference-field energy, never divide by a near-zero reference shell. Exclude the DC coefficient from the spatial-wavenumber axis and state this in the caption. Use logarithmic y scale, full available nonzero spatial shells, and no smoothing. No spectral confidence or significance claim. Burgers transforms only its spatial axis; nonperiodic fields use DCT-II; periodic fields use FFT.

Palette: truth #2B2B2B, single prediction #246591, average #C99732. Distinguish lines by dashes as well as color. Label axes, transform convention, sample count and normalization in captions. Inspect actual PNGs and embedded fonts, then inspect the combined manuscript PDF after final insertion.

2026-09-08 图形调整：Wave 的 a 场效应明显大于其余场，仍保留覆盖全部区间的线性横轴。右侧逐行标注平均误差变化（百分点），让较小的效应也可直接读数；不截断 Wave 区间，不改变统计指标或剔除行。

Numeric labels: retain two significant digits for small nonzero errors and changes; do not display them as 0.00. Steady Heat has errors below 0.005%, so use decimal or scientific notation as needed.

Full-study visual QA: Shallow Water has an isolated final-shell reference energy near 1e-38. A purely logarithmic axis compresses the rest of its spectrum. Use a common symmetric-log scale with a linear region below 1e-14 of full reference energy, retaining every point and stating the scale in the captions. This changes only display geometry; no shell value is clipped, floored, smoothed or removed. Small ensemble-table rows use four decimal places where needed to distinguish their means and SDs.
