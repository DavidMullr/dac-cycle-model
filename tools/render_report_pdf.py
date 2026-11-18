"""Render the concise current report to a standalone review PDF."""
from importlib.metadata import distribution
import os
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'model/report/project'

override = os.environ.get("DAC_REPORT_FONTS")
font_dir = (Path(override) if override else
            Path(distribution("matplotlib").locate_file("matplotlib/mpl-data/fonts/ttf")))
font_files = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf")
for name, filename in zip(('Report', 'ReportBold', 'ReportItalic'), font_files):
    pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold',italic='ReportItalic',boldItalic='ReportBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportBody',fontName='Report',fontSize=9.5,leading=13.2,spaceAfter=7,textColor=colors.HexColor('#253b42')))
styles.add(ParagraphStyle(name='ReportTitle',fontName='ReportBold',fontSize=23,leading=27,spaceAfter=7,textColor=colors.HexColor('#17353e')))
styles.add(ParagraphStyle(name='ReportSub',fontName='Report',fontSize=10,leading=14,spaceAfter=14,textColor=colors.HexColor('#536970')))
styles.add(ParagraphStyle(name='ReportHeading',fontName='ReportBold',fontSize=13,leading=17,spaceBefore=13,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#176e68')))
styles.add(ParagraphStyle(name='ReportCell',fontName='Report',fontSize=8.3,leading=11,spaceAfter=0))
styles.add(ParagraphStyle(name='ReportCellHead',parent=styles['ReportCell'],fontName='ReportBold',textColor=colors.HexColor('#17353e')))

def inline(text):
    text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',lambda m: m.group(1),text)
    text=escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',text)
    text=re.sub(r'`([^`]+)`',r'\1',text)
    return text.replace('\u2013','-').replace('\u2014','-')

raw=(REPORT/'index.qmd').read_text(encoding='utf-8')
raw=re.sub(r'^---.*?---\s*','',raw,flags=re.S)
raw=raw.split('[Result table and exact exports]')[0]
raw=re.sub(r'^\[Download the four-page report.*?\n', '', raw, flags=re.M)
flow=[Paragraph('Direct-air-capture model and results',styles['ReportTitle']),
      Paragraph('Carbon Team \u00b7 Ast\u00e9rix | Technical review | 17 November 2026',styles['ReportSub'])]
lines=raw.splitlines();i=0;width=A4[0]-36*mm
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('## '):
        flow.append(Paragraph(inline(line[3:]),styles['ReportHeading']));i+=1;continue
    if line.startswith('|'):
        matrix=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[s.strip() for s in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',c) for c in cells):
                matrix.append([Paragraph(inline(c),styles['ReportCellHead' if not matrix else 'ReportCell']) for c in cells])
            i+=1
        widths=[width*.39,width*.61] if len(matrix[0])==2 else [width*.43,width*.285,width*.285]
        table=Table(matrix,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1ee')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#91b7ae')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8f7')])]))
        flow.extend([table,Spacer(1,8)]);continue
    paragraph=[]
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('## ','|')):
        paragraph.append(lines[i].strip());i+=1
    flow.append(KeepTogether([Paragraph(inline(' '.join(paragraph)),styles['ReportBody'])]))

def page(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c9d8d4'))
    canvas.line(18*mm,16*mm,A4[0]-18*mm,16*mm)
    canvas.setFont('Report',8);canvas.setFillColor(colors.HexColor('#536970'))
    canvas.drawString(18*mm,11*mm,'Carbon Team | Conditional model results')
    canvas.drawRightString(A4[0]-18*mm,11*mm,str(doc.page))
    canvas.restoreState()

target=REPORT/'model_and_results.pdf'
doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,
                     topMargin=16*mm,bottomMargin=23*mm,title='Carbon Team: DAC model and results',author='Carbon Team')
doc.build(flow,onFirstPage=page,onLaterPages=page)
print(target)
