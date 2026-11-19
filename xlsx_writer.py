"""
A small .xlsx writer using only the standard library, so the grading
bundle needs nothing beyond what the students' own code needs.

    write_xlsx({"Marks": (headers, rows), ...}, "out.xlsx")

Bold frozen header, autofilter, numbers kept numeric, text inline.
"""
import zipfile
from xml.sax.saxutils import escape


def _col(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _cell(ref, v, style=0):
    st = f' s="{style}"' if style else ""
    if v is None or (isinstance(v, float) and v != v):
        return f'<c r="{ref}"{st}/>'
    if isinstance(v, bool):
        v = "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f'<c r="{ref}"{st}><v>{v}</v></c>'
    t = escape(str(v)).replace("\x00", "")
    t = "".join(ch for ch in t if ch in "\t\n\r" or ord(ch) >= 32)
    return f'<c r="{ref}" t="inlineStr"{st}><is><t xml:space="preserve">{t}</t></is></c>'


def _sheet(headers, rows):
    widths = [min(60, max([len(str(h))] + [len(str(r[i])) for r in rows[:500]]) + 2)
              for i, h in enumerate(headers)]
    cols = "".join(f'<col min="{i+1}" max="{i+1}" width="{w}" customWidth="1"/>'
                   for i, w in enumerate(widths))
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
           '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
           'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>']
    if headers:
        out.append(f"<cols>{cols}</cols>")
    out.append("<sheetData>")
    out.append('<row r="1">' + "".join(_cell(f"{_col(i)}1", h, 1)
                                       for i, h in enumerate(headers)) + "</row>")
    for r, row in enumerate(rows, start=2):
        out.append(f'<row r="{r}">' + "".join(_cell(f"{_col(i)}{r}", v)
                                              for i, v in enumerate(row)) + "</row>")
    out.append("</sheetData>")
    if headers:
        out.append(f'<autoFilter ref="A1:{_col(len(headers)-1)}{max(1, len(rows)+1)}"/>')
    out.append("</worksheet>")
    return "".join(out)


def write_xlsx(sheets, path):
    names = list(sheets)
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + "".join(f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                      for i in range(len(names))) + "</Types>")
        z.writestr("_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>")
        z.writestr("xl/workbook.xml",
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>'
            + "".join(f'<sheet name="{escape(n[:31])}" sheetId="{i+1}" r:id="rId{i+1}"/>'
                      for i, n in enumerate(names)) + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(f'<Relationship Id="rId{i+1}" Type="{rel}/worksheet" Target="worksheets/sheet{i+1}.xml"/>'
                      for i in range(len(names)))
            + f'<Relationship Id="rId{len(names)+1}" Type="{rel}/styles" Target="styles.xml"/>'
            "</Relationships>")
        z.writestr("xl/styles.xml",
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="{ns}">'
            '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
            '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
            '<cellXfs count="2"><xf fontId="0"/><xf fontId="1" applyFont="1"/></cellXfs>'
            "</styleSheet>")
        for i, n in enumerate(names):
            headers, rows = sheets[n]
            z.writestr(f"xl/worksheets/sheet{i+1}.xml", _sheet(list(headers), [list(r) for r in rows]))
