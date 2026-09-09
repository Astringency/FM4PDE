# Paper archive and rebuild review

The current archive/restore implementation and documented rebuild sequence passed this independent read-only review. All 107 direct figure/input references from the four TeX documents were present and selected for capture. The bibliography, JMLR style, local TeX sources, numerical tables, review material, audit code and four PDFs are included. The paper tree contained no symbolic links.

The review identified and the root agent corrected four issues: restoration into an archive subdirectory, verification of the copied destination, binding saved QA to its actual TeX/PDF/figure bytes, and an explicit final-manuscript completion gate. The final gate appropriately remains distinct from completion of server result transfers. The README rebuilds a fresh copy and verifies the unchanged original archive afterwards.

The recorded relocation test restored 1,370 files (335,617,106 bytes), built all four documents, and obtained identical extracted layout text at 185, 185, 51 and 13 pages. The original archive retained its verified manifest and file hashes. That test covered the current NS-integrated working draft; it does not certify completion of the separate K experiment or the final paper version.

Compilation regenerates auxiliary and PDF files in the restored tree. Their bytes need not match the archived PDFs, whose unchanged hashes are checked separately. Numerical reproduction of figure data uses STUDY_RECIPES.md and the raw result archive; rebuilding TeX alone does not recompute predictions or statistics. Exact chart rendering additionally uses the four recorded Times New Roman files through FM4PDE_FONT_DIR.

The reviewed script, README and rebuild-receipt hashes are in PAPER_ASSETS_REVIEW.json. No canonical paper file or root-owned archive tool was changed by this review.
