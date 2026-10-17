"""T8.2 / T9.10: build the IEEE-format report from reports/report.md.

    python scripts/build_report.py

1. Checks every quoted number against the tables (check_report_numbers.py);
   the build stops if any is flagged.
2. pandoc report.md -> reports/MirageAI_Report.html, using the IEEE conference
   template in reports/ieee/ (two columns, A4, Times, numbered sections).
3. Headless Chrome or Edge prints the HTML to MirageAI_Report.pdf, and the page
   count is reported. No LaTeX engine or Word is needed.
4. pandoc also writes MirageAI_Report.docx, an editable Word copy in the same
   IEEE layout: styles from a generated reference document, two columns and
   section numbering from reports/ieee/ieee-docx.lua.

pandoc is looked up on PATH, then in its default per-user install location
(`winget install JohnMacFarlane.Pandoc`). The browser is looked up on PATH and
in the default Chrome / Edge install locations.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SRC = REPORTS / "report.md"
HTML = REPORTS / "MirageAI_Report.html"
PDF = REPORTS / "MirageAI_Report.pdf"
DOCX = REPORTS / "MirageAI_Report.docx"
IEEE = REPORTS / "ieee"
FROM = "markdown+pipe_tables+table_captions+implicit_figures+fenced_divs+tex_math_dollars"

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def pandoc() -> str:
    found = shutil.which("pandoc")
    if found:
        return found
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Pandoc" / "pandoc.exe"
    if local.exists():
        return str(local)
    sys.exit("pandoc not found: install it with `winget install JohnMacFarlane.Pandoc`")


def browser() -> str:
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for path in BROWSERS:
        if Path(path).exists():
            return path
    sys.exit("Chrome or Edge not found: it is needed to print the report to PDF")


def _pstyle(sid: str, name: str, ppr: str = "", rpr: str = "", based: str = "Normal", extra: str = "") -> str:
    return (f'<w:style w:type="paragraph" w:customStyle="1" w:styleId="{sid}"><w:name w:val="{name}"/>'
            f'<w:basedOn w:val="{based}"/>{extra}<w:qFormat/><w:pPr>{ppr}</w:pPr><w:rPr>{rpr}</w:rPr></w:style>')


TNR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
SZ = '<w:sz w:val="{0}"/><w:szCs w:val="{0}"/>'.format
COLUMN = 5021  # one column of the two-column body, twips

# Word styles for the IEEE layout: Times 10 pt justified body, small-caps section
# headings, 8 pt captions, tables and references. Replaces pandoc's defaults.
DOCX_STYLES = {
    "Normal": ('<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/>'
               '<w:pPr><w:spacing w:after="0" w:line="228" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr></w:style>'),
    "BodyText": _pstyle("BodyText", "Body Text", '<w:ind w:firstLine="202"/>'),
    "FirstParagraph": _pstyle("FirstParagraph", "First Paragraph", based="BodyText", extra='<w:next w:val="BodyText"/>'),
    "Compact": _pstyle("Compact", "Compact", '<w:spacing w:before="0" w:after="0"/><w:jc w:val="left"/>'),
    "Title": _pstyle("Title", "Title", '<w:spacing w:before="0" w:after="200" w:line="240" w:lineRule="auto"/>'
                     '<w:jc w:val="center"/>', SZ(48)),
    "Heading1": _pstyle("Heading1", "heading 1", '<w:keepNext/><w:keepLines/><w:spacing w:before="160" w:after="80"/>'
                        '<w:jc w:val="center"/><w:outlineLvl w:val="0"/>', "<w:smallCaps/>" + SZ(20),
                        extra='<w:next w:val="BodyText"/>'),
    "Heading2": _pstyle("Heading2", "heading 2", '<w:keepNext/><w:keepLines/><w:spacing w:before="120" w:after="60"/>'
                        '<w:jc w:val="left"/><w:outlineLvl w:val="1"/>', "<w:i/>" + SZ(20),
                        extra='<w:next w:val="BodyText"/>'),
    "Abstract": _pstyle("Abstract", "Abstract", '<w:spacing w:after="100"/><w:ind w:firstLine="202"/>', "<w:b/>" + SZ(18)),
    "AuthorBlock": _pstyle("AuthorBlock", "Author Block", '<w:jc w:val="center"/>'),
    "Equation": _pstyle("Equation", "Equation", f'<w:tabs><w:tab w:val="center" w:pos="{COLUMN // 2}"/>'
                        f'<w:tab w:val="right" w:pos="{COLUMN}"/></w:tabs><w:spacing w:before="60" w:after="60"/>'
                        '<w:jc w:val="left"/>'),
    "Algorithm": _pstyle("Algorithm", "Algorithm", '<w:pBdr><w:top w:val="single" w:sz="6" w:space="2" w:color="000000"/>'
                         '<w:bottom w:val="single" w:sz="6" w:space="2" w:color="000000"/></w:pBdr>'
                         '<w:spacing w:before="0" w:after="0"/><w:jc w:val="left"/>', SZ(18)),
    "Reference": _pstyle("Reference", "Reference", '<w:ind w:left="340" w:hanging="340"/><w:spacing w:after="20"/>', SZ(16)),
    "Caption": _pstyle("Caption", "caption", '<w:spacing w:before="60" w:after="160"/>', SZ(16)),
    "ImageCaption": _pstyle("ImageCaption", "Image Caption", based="Caption"),
    "TableCaption": _pstyle("TableCaption", "Table Caption", '<w:keepNext/><w:jc w:val="center"/>'
                            '<w:spacing w:before="160" w:after="60"/>', based="Caption"),
    "Figure": _pstyle("Figure", "Figure", '<w:spacing w:before="120" w:after="0"/><w:jc w:val="center"/>'),
    "CaptionedFigure": _pstyle("CaptionedFigure", "Captioned Figure", "<w:keepNext/>", based="Figure"),
    "Table": ('<w:style w:type="table" w:default="1" w:styleId="Table"><w:name w:val="Table"/>'
              '<w:basedOn w:val="TableNormal"/><w:qFormat/><w:pPr><w:jc w:val="left"/></w:pPr><w:rPr>' + SZ(16)
              + '</w:rPr><w:tblPr><w:jc w:val="center"/><w:tblBorders>'
              '<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
              '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/></w:tblBorders>'
              '<w:tblCellMar><w:top w:w="15" w:type="dxa"/><w:left w:w="60" w:type="dxa"/>'
              '<w:bottom w:w="15" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar></w:tblPr>'
              '<w:tblStylePr w:type="firstRow"><w:rPr><w:b/></w:rPr><w:tcPr><w:tcBorders>'
              '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tcBorders>'
              '<w:vAlign w:val="bottom"/></w:tcPr></w:tblStylePr></w:style>'),
    "VerbatimChar": ('<w:style w:type="character" w:customStyle="1" w:styleId="VerbatimChar"><w:name w:val="Verbatim Char"/>'
                     '<w:basedOn w:val="DefaultParagraphFont"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
                     '</w:rPr></w:style>'),
    "Hyperlink": ('<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/>'
                  '<w:basedOn w:val="DefaultParagraphFont"/><w:rPr><w:color w:val="1F3A93"/></w:rPr></w:style>'),
}
DOCX_DEFAULTS = (f"<w:docDefaults><w:rPrDefault><w:rPr>{TNR}{SZ(20)}"
                 '<w:lang w:val="en-GB" w:eastAsia="en-US" w:bidi="ar-SA"/></w:rPr></w:rPrDefault>'
                 '<w:pPrDefault><w:pPr><w:spacing w:after="0"/></w:pPr></w:pPrDefault></w:docDefaults>')
# A4, IEEE conference margins, two columns; the filter's section breaks repeat this.
DOCX_SECTION = ('<w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="11906" w:h="16838"/>'
                '<w:pgMar w:top="1077" w:right="812" w:bottom="1361" w:left="812" w:header="567" w:footer="680"'
                ' w:gutter="0"/><w:cols w:num="2" w:space="239"/></w:sectPr>')


def ieee_reference_docx(dest: Path) -> None:
    """pandoc's default reference.docx with the IEEE styles and page setup."""
    raw = subprocess.run([pandoc(), "--print-default-data-file", "reference.docx"],
                         check=True, capture_output=True).stdout
    with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/styles.xml":
                xml = re.sub(r"<w:docDefaults>.*?</w:docDefaults>", DOCX_DEFAULTS, data.decode("utf-8"), flags=re.S)
                for sid in DOCX_STYLES:
                    xml = re.sub(rf'<w:style [^>]*w:styleId="{sid}">.*?</w:style>', "", xml, flags=re.S)
                # drop the theme fonts and colours of the remaining heading styles
                xml = re.sub(r"<w:rFonts w:asciiTheme[^>]*/>|<w:color [^>]*/>", "", xml)
                data = xml.replace("</w:styles>", "".join(DOCX_STYLES.values()) + "</w:styles>").encode("utf-8")
            elif item.filename == "word/document.xml":
                data = re.sub(r"<w:sectPr>.*?</w:sectPr>", DOCX_SECTION, data.decode("utf-8"), flags=re.S).encode()
            out.writestr(item, data)


def finish_docx(path: Path) -> None:
    """Drop pandoc's author paragraphs (the filter writes an author table) and
    narrow the list indents to suit a 3.5-inch column."""
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/document.xml":
                xml = re.sub(r'<w:p>(?:(?!</w:p>).)*?<w:pStyle w:val="Author" />.*?</w:p>', "",
                             data.decode("utf-8"), flags=re.S)
                # keep each table on one page/column
                xml = re.sub(r"<w:tbl>.*?</w:tbl>",
                             lambda m: re.sub(r"(<w:pPr>\s*<w:pStyle [^>]*/>)", r"\1<w:keepNext/>", m[0]),
                             xml, flags=re.S)
                data = xml.encode("utf-8")
            elif item.filename == "word/numbering.xml":
                data = re.sub(r'w:left="(\d+)" w:hanging="(\d+)"',
                              lambda m: f'w:left="{int(m[1]) // 2}" w:hanging="{int(m[2]) // 2}"',
                              data.decode("utf-8")).encode("utf-8")
            out.writestr(item, data)
    tmp.replace(path)


def page_count(pdf: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page(?![s\w])", pdf.read_bytes()))


def main() -> int:
    check = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_report_numbers.py")])
    if check.returncode:
        sys.exit("number check failed; fix report.md before building")

    subprocess.run([pandoc(), str(SRC), "-o", str(HTML), "--from", FROM, "--standalone", "--math-method=mathml",
                    "--template", str(IEEE / "template.html"), "--css", "ieee/ieee.css"],
                   check=True, cwd=REPORTS)
    print(f"wrote {HTML.relative_to(ROOT)}")

    PDF.unlink(missing_ok=True)
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
                    f"--print-to-pdf={PDF}", HTML.as_uri()],
                   check=True, capture_output=True, timeout=180)
    if not PDF.exists():
        sys.exit("the browser did not write the PDF")
    print(f"wrote {PDF.relative_to(ROOT)}  [pages={page_count(PDF)}]")

    with tempfile.TemporaryDirectory() as tmp:
        reference = Path(tmp) / "reference.docx"
        ieee_reference_docx(reference)
        subprocess.run([pandoc(), str(SRC), "-o", str(DOCX), "--from", FROM, "--resource-path", str(REPORTS),
                        "--reference-doc", str(reference), "--lua-filter", str(IEEE / "ieee-docx.lua")],
                       check=True, cwd=REPORTS)
    finish_docx(DOCX)
    print(f"wrote {DOCX.relative_to(ROOT)} (editable, IEEE two-column)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
