#!/usr/bin/env python3
"""Markdown -> PDF, pure Python. No LaTeX, no system libraries.

    python3 md2pdf.py PRESENTATION-SCRIPTS.md

weasyprint would give nicer output but needs pango/cairo installed system-wide.
xhtml2pdf is reportlab-based and pure Python, so this runs anywhere.

Unicode matters here: the document contains Rs (U+20B9), micro, minus, gamma,
times and middot. ReportLab's built-in fonts are latin-1 only, so a Unicode TTF
is registered first. Any character the chosen font lacks is transliterated
rather than silently dropped as a black box.
"""
import os, sys, glob, re

import markdown
from xhtml2pdf import pisa
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# DejaVu FIRST, not Arial Unicode. Arial Unicode predates the Indian Rupee
# sign (U+20B9, added to Unicode in 2010) and drops it silently -- which is
# fatal for a document whose every cost figure is in rupees. Verified with
# fontTools: DejaVu has it, Arial Unicode and Verdana and Tahoma do not.
FONT_CANDIDATES = []
try:
    import matplotlib
    FONT_CANDIDATES += [os.path.join(os.path.dirname(matplotlib.__file__),
                        "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")]
except Exception:
    pass
FONT_CANDIDATES += [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]

RUPEE = 0x20B9


def font_has_rupee(path):
    """Only accept a font that can actually render the rupee sign."""
    try:
        from fontTools.ttLib import TTFont as _FT
        f = _FT(path, fontNumber=0, lazy=True)
        return any(RUPEE in t.cmap for t in f["cmap"].tables)
    except Exception:
        return True          # cannot check -- do not block on it


def register_font():
    for require_rupee in (True, False):          # prefer a font with Rs.
        for p in FONT_CANDIDATES:
            if not os.path.exists(p):
                continue
            if require_rupee and not font_has_rupee(p):
                continue
            try:
                pdfmetrics.registerFont(TTFont("Doc", p))
                bold = p.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
                if os.path.exists(bold) and bold != p:
                    pdfmetrics.registerFont(TTFont("Doc-Bold", bold))
                    pdfmetrics.registerFontFamily(
                        "Doc", normal="Doc", bold="Doc-Bold",
                        italic="Doc", boldItalic="Doc-Bold")
                return "Doc", p
            except Exception:
                continue
    return "Helvetica", None


CSS = """
@page { size: A4; margin: 1.7cm 1.6cm 1.8cm 1.6cm; }
body, p, li, td, th, table, h1, h2, h3, strong, b, em, blockquote, div
     { font-family: %(font)s; }
body { font-size: 9.4pt; line-height: 1.42; color: #16202b; }
h1 { font-size: 17pt; color: #0f6f80; margin: 16pt 0 6pt; }
h2 { font-size: 12.5pt; color: #16202b; margin: 13pt 0 4pt;
     border-bottom: 0.6pt solid #c8d2dd; padding-bottom: 2pt; }
h3 { font-size: 10.4pt; color: #35485c; margin: 9pt 0 3pt; }
p  { margin: 0 0 5pt; }
ul, ol { margin: 0 0 6pt 12pt; }
li { margin-bottom: 2pt; }
strong { color: #0b1620; }
code { font-family: Courier; font-size: 8.6pt; background: #eef1f5; }
table { width: 100%%; border-collapse: collapse; margin: 5pt 0 9pt;
        font-size: 8.6pt; }
th { background: #0f6f80; color: #ffffff; text-align: left;
     padding: 3.5pt 5pt; font-size: 8pt; }
td { border-bottom: 0.5pt solid #d8e0e8; padding: 3.5pt 5pt;
     vertical-align: top; }
hr { border: 0; border-top: 0.6pt solid #c8d2dd; margin: 11pt 0; }
blockquote { margin: 4pt 0 6pt 10pt; color: #47586b; }
"""

# Characters ReportLab's latin-1 fonts cannot show. Only used if no Unicode
# font was found -- better a readable transliteration than a black box.
FALLBACK = {"₹": "Rs.", "µ": "u", "−": "-", "×": "x",
            "≈": "~", "γ": "gamma", "²": "2", "·": "-",
            "→": "->", "≤": "<=", "≥": ">=", "‘": "'",
            "’": "'", "“": '"', "”": '"', "–": "-",
            "—": "--", "…": "..."}


def convert(src, dst=None):
    dst = dst or os.path.splitext(src)[0] + ".pdf"
    font, path = register_font()
    print("  font: %s%s" % (font, " (%s)" % os.path.basename(path) if path else
                            "  -- latin-1 only, transliterating"))

    text = open(src, encoding="utf-8").read()
    if font == "Helvetica":
        for a, b in FALLBACK.items():
            text = text.replace(a, b)

    body = markdown.markdown(text, extensions=["tables", "fenced_code",
                                               "sane_lists"])
    # xhtml2pdf resolves fonts through @font-face in the CSS, not through
    # reportlab registration alone -- without this the rupee sign renders as a
    # black box even though the font contains it.
    face = ""
    if path:
        face = ('@font-face { font-family: "Doc"; src: url("%s"); }\n' % path)
        bold = path.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
        if os.path.exists(bold) and bold != path:
            face += ('@font-face { font-family: "Doc"; src: url("%s"); '
                     'font-weight: bold; }\n' % bold)
    html = ("<html><head><meta charset='utf-8'><style>%s%s</style></head>"
            "<body>%s</body></html>" % (face, CSS % {"font": font}, body))

    with open(dst, "wb") as fh:
        res = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if res.err:
        sys.exit("  conversion failed with %d errors" % res.err)
    print("  %s -> %s  (%.0f KB)" % (src, dst, os.path.getsize(dst) / 1024))
    return dst


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: md2pdf.py <file.md> [out.pdf]")
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
