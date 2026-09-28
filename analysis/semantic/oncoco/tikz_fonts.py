"""Shared font scaling for the generated TikZ figures.

The figures are laid out at 17.4 cm (hero, noise band) and 13.05 cm (HMM) and
are included at that width in the paper, so a point in the figure is a point on
the page. The original label sizes (4.7 to 6.6 pt) were too small to read in
print. scale_fonts() multiplies every \\fontsize{a}{b} in the emitted TeX by
FONT_SCALE and maps the two LaTeX size commands used for titles to fixed sizes.
"""
import re

FONT_SCALE = 1.2
TITLE = r"\fontsize{9.5}{11}\selectfont"    # was \small (9 pt)
SUBTITLE = r"\fontsize{7.8}{9.2}\selectfont"  # was \scriptsize (7 pt)


def scale_fonts(tex: str, factor: float = FONT_SCALE) -> str:
    def _scale(m: re.Match) -> str:
        a = float(m.group(1)) * factor
        b = float(m.group(2)) * factor
        return rf"\fontsize{{{a:.1f}}}{{{b:.1f}}}"

    tex = re.sub(r"\\fontsize\{([\d.]+)\}\{([\d.]+)\}", _scale, tex)
    return tex.replace(r"\small", TITLE).replace(r"\scriptsize", SUBTITLE)
