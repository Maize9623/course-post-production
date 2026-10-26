#!/usr/bin/env python3
"""Native course-brand artwork with Pillow; no footage or network dependencies.

Python 3.9+ and Pillow are required. All coordinates are native 2560x1440.
Diagram positions, labels and relationship types must be supplied explicitly.
This renderer wraps text geometrically; it never invents teaching relationships.

Examples:
  python render_brand.py --cover --title '用 AI 整理资料' \
    --subtitle '从原始材料到可核验的结果' --speaker 讲师 --out cover.png
  python render_brand.py --diagram diagram.json --highlight memory --out diagram.png
  python render_brand.py --diagram practical.json --profile practical --out practical.png
  python render_brand.py --diagram lesson.json --project-style project-style.json --out lesson.png
  python render_brand.py --caption '先确认材料，再检查结果' --out subtitle.png
  python render_brand.py --transition --title '进入实践' --preview --out-dir frames

Transition PNGs are full RGB and opaque, including frames 0 and 77. The editor
must hide the lecturer and captions in this 78-frame core AND in the 18-frame
guards immediately before and after it. Background content may fade in those
guards. Any lecturer fade-out must finish BEFORE the preceding guard; any
fade-in may begin only AFTER the following guard. Never composite a lecturer
or captions above the core or either guard. This script renders the core only;
the external guard timing and layer suppression are owned by the editor.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

# The installed skill's style file is canonical; matching defaults let this one
# script run independently when copied without the surrounding skill package.
STYLE_PATH = Path(__file__).resolve().parents[1]/'assets'/'style.json'
STYLE = json.loads(STYLE_PATH.read_text(encoding='utf-8')) if STYLE_PATH.is_file() else {}
_COLORS = STYLE.get('colors',{})
_FONT = STYLE.get('font',{})
_LAYOUT = STYLE.get('layout',{})
_CANVAS = STYLE.get('canvas',{})
_TRANSITIONS = STYLE.get('transitions',{})
WIDTH, HEIGHT, FPS = (int(_CANVAS.get(k,v)) for k,v in (('width',2560),('height',1440),('fps',60)))
TRANSITION_FRAMES = int(_TRANSITIONS.get('frames',78))
TRANSITION_SECONDS = float(_TRANSITIONS.get('duration_seconds',1.3))
BG = _COLORS.get('background','#F5F7F5')
INK = _COLORS.get('ink','#192523')
ACCENT = _COLORS.get('accent','#00877F')
MUTED = _COLORS.get('muted','#75817C')
BORDER = _COLORS.get('line','#D6DEDA')
CARD = _COLORS.get('paper','#FFFFFF')
HIGHLIGHT = _COLORS.get('focus_fill','#EAF5F1')
SERIES = STYLE.get('series','知识课程')
TITLE_SIZE = int(_FONT.get('section_title_px',120))
DIAGRAM_TITLE_SIZE = int(_FONT.get('lesson_title_px',62))
DIAGRAM_NODE_SIZE = int(_FONT.get('node_title_px',40))
SUBTITLE_SIZE = int(_FONT.get('subtitle_px',54))
DIAGRAM_FONT_MIN, DIAGRAM_FONT_MAX = 36, 52
CAPTION_HEIGHT = 160
CAPTION_TOP = HEIGHT-CAPTION_HEIGHT
_lecturer = _LAYOUT.get('lecturer_circle',[56,945,340])
LECTURER_RECT = (_lecturer[0],_lecturer[1],_lecturer[0]+_lecturer[2],_lecturer[1]+_lecturer[2])
# Nodes and edges keep an additional 24px gap around the lecturer window.
LECTURER_CLEAR = (LECTURER_RECT[0]-24,LECTURER_RECT[1]-24,LECTURER_RECT[2]+24,CAPTION_TOP)
CONTENT_PANEL = tuple(_LAYOUT.get('content_panel',[410,264,2480,1246]))
DIAGRAM_AREA = (CONTENT_PANEL[0]+24,320,CONTENT_PANEL[2]-24,CONTENT_PANEL[3]-24)
FONT_CANDIDATES = (
    _FONT.get('preferred_path'),
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/adobe-source-han-sans/SourceHanSansSC-Regular.otf',
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    'C:/Windows/Fonts/msyh.ttc',
)


class LayoutError(ValueError):
    """Invalid geometry, unsupported text fit, or a violated reserved area."""


def layout_settings(profile=None, project_style=None):
    """Resolve layout without mutating module defaults or another render call."""
    if profile and project_style:
        raise LayoutError('Choose either profile or project_style, not both.')
    if project_style:
        settings = json.loads(Path(project_style).read_text(encoding='utf-8-sig'))
        layout = settings.get('layout', {})
        selected = settings.get('profile', 'project')
    else:
        selected = STYLE.get('mode_profiles', {}).get(profile or 'theory', profile or 'theory')
        profiles = STYLE.get('profiles', {})
        if selected not in profiles and selected != 'theory':
            raise LayoutError(f'Unknown brand profile: {profile}')
        layout = {**_LAYOUT, **profiles.get(selected, {}).get('layout', {})}
    canvas = settings.get('canvas', {}) if project_style else _CANVAS
    if any(float(canvas.get(key, default)) != default for key, default in (('width', WIDTH), ('height', HEIGHT), ('fps', FPS))):
        raise LayoutError('Project style must keep the series 2560x1440, 60 fps canvas.')
    circle = layout.get('lecturer_circle', [56, 945, 340])
    if len(circle) != 3 or any(not math.isfinite(float(value)) for value in circle) or circle[2] <= 0:
        raise LayoutError('lecturer_circle must be [left, top, positive diameter] in native pixels.')
    panel = tuple(layout.get('content_panel', CONTENT_PANEL))
    checked_box(panel, name='Content panel')
    area = (panel[0]+24, max(320, panel[1]+56), panel[2]-24, panel[3]-24)
    checked_box(area, name='Diagram area')
    reserve = (circle[0]-24, circle[1]-24, circle[0]+circle[2]+24, min(CAPTION_TOP, circle[1]+circle[2]+24))
    return dict(profile=selected, layout=layout, panel=panel, area=area, lecturer_clear=reserve)


@lru_cache(maxsize=32)
def font(size: int, font_path: Optional[str] = None):
    path = font_path or os.environ.get('COURSE_BRAND_FONT')
    if path:
        if not Path(path).is_file():
            raise LayoutError(f'Font file does not exist: {path}')
    else:
        path = next((p for p in FONT_CANDIDATES if p and Path(p).is_file()),None)
        if not path and shutil.which('fc-match'):
            match = subprocess.run(['fc-match','-f','%{file}\n','sans:lang=zh-cn'],capture_output=True,text=True)
            path = next((p for p in match.stdout.splitlines() if Path(p).is_file()),None)
    if not path:
        raise LayoutError('A suitable font is required. Install Noto Sans CJK/Source Han Sans, pass --font, or set COURSE_BRAND_FONT.')
    return ImageFont.truetype(str(path),int(size),index=int(_FONT.get('regular_index',0)))


def ensure_glyphs(face, text):
    """Reject missing non-ASCII glyphs instead of silently drawing tofu."""
    missing = face.getmask(chr(0x10FFFF))
    signature = (missing.size, bytes(missing))
    for character in set(str(text)):
        if ord(character) < 128 or character.isspace():
            continue
        mask = face.getmask(character)
        if (mask.size, bytes(mask)) == signature:
            raise LayoutError(f'Font lacks glyph {character!r}; use --font with a font covering this language.')


def smooth(value: float) -> float:
    value=max(0.,min(1.,float(value)))
    return value*value*(3.-2.*value)


def overlap(a,b):
    return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]


def checked_box(box, *, name, allowed=(0,0,WIDTH,CAPTION_TOP), lecturer_clear=False, lecturer_reserve=None):
    if len(box)!=4 or any(not math.isfinite(float(x)) for x in box):
        raise LayoutError(f'{name}: box must contain four finite coordinates.')
    x0,y0,x1,y1=map(float,box)
    if not(x0<x1 and y0<y1):raise LayoutError(f'{name}: box has non-positive size.')
    if x0<allowed[0] or y0<allowed[1] or x1>allowed[2] or y1>allowed[3]:
        raise LayoutError(f'{name}: box {tuple(box)} exceeds allowed area {allowed}.')
    if lecturer_clear and overlap(box,LECTURER_CLEAR if lecturer_reserve is None else lecturer_reserve):
        raise LayoutError(f'{name}: content overlaps the selected profile\'s lecturer area.')
    return tuple(box)


def tokens(text):
    """Keep Latin words together; Chinese glyphs remain natural wrap points."""
    import re
    return re.findall(r'[A-Za-z0-9_][A-Za-z0-9_./:+@#%-]*|[^A-Za-z0-9_]',text)


def wrap_text(text, face, width, max_lines):
    lines=[]
    for paragraph in str(text).split('\n'):
        current=''
        if not paragraph:
            lines.append('');continue
        for token in tokens(paragraph):
            if face.getlength(token)>width:
                raise LayoutError(f'Unbreakable text exceeds available width: {token!r}')
            trial=current+token
            if current and face.getlength(trial)>width:
                lines.append(current.rstrip());current=token.lstrip()
            else:current=trial
        lines.append(current.rstrip())
    if len(lines)>max_lines:
        raise LayoutError(f'Text needs {len(lines)} lines; only {max_lines} allowed. Enlarge its box or shorten the text.')
    return lines


def text_plan(text, box, size, *, max_lines=1, line_gap=12, align='left', valign='center', font_path=None):
    face=font(size,font_path);ensure_glyphs(face,text);width=box[2]-box[0]
    lines=wrap_text(text,face,width,max_lines)
    # The real glyph bounds, including Latin descenders, determine placement.
    metrics=[face.getbbox(line or ' ') for line in lines]
    heights=[max(1,b[3]-b[1]) for b in metrics]
    total=sum(heights)+line_gap*max(0,len(lines)-1)
    if total>box[3]-box[1]:raise LayoutError(f'Text height {total}px exceeds {box[3]-box[1]}px at {size}px font.')
    top=box[1] if valign=='top' else box[1]+(box[3]-box[1]-total)/2
    result=[]
    for line,b,height in zip(lines,metrics,heights):
        actual=b[2]-b[0]
        if actual>width:raise LayoutError(f'Ink width exceeds box at {size}px: {line!r}')
        left=box[0] if align=='left' else box[0]+(width-actual)/2
        origin=(round(left-b[0]),round(top-b[1]))
        bounds=(origin[0]+b[0],origin[1]+b[1],origin[0]+b[2],origin[1]+b[3])
        if bounds[0]<box[0]-1 or bounds[2]>box[2]+1 or bounds[1]<box[1]-1 or bounds[3]>box[3]+1:
            raise LayoutError(f'Measured text bounds exceed box: {line!r}')
        result.append(dict(text=line,origin=origin,bounds=bounds));top+=height+line_gap
    return dict(size=size,box=box,lines=result,font=face)


def draw_plan(draw, plan, *, fill=INK, stroke_width=0, stroke_fill=None):
    for row in plan['lines']:
        draw.text(row['origin'],row['text'],font=plan['font'],fill=fill,
            stroke_width=stroke_width,stroke_fill=stroke_fill)


def render_cover(title, subtitle='', speaker='讲师', *, series=SERIES, label='', font_path=None):
    """RGB cover. No photo, footer, progress bar, or production label is added."""
    if not str(title).strip():raise LayoutError('Cover title must not be empty.')
    im=Image.new('RGB',(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(im)
    draw_plan(draw,text_plan(series,(168,120,1980,190),48,font_path=font_path),fill=MUTED)
    if speaker:
        plan=text_plan(speaker,(2050,120,2392,190),42,align='center',font_path=font_path)
        draw_plan(draw,plan,fill=MUTED)
    if label:
        draw_plan(draw,text_plan(label,(400,295,2360,360),40,font_path=font_path),fill=ACCENT)
    title_box=(400,420,2392,765)
    title_plan=text_plan(title,title_box,TITLE_SIZE,max_lines=2,line_gap=28,font_path=font_path)
    draw_plan(draw,title_plan)
    title_top=min(row['bounds'][1] for row in title_plan['lines'])
    title_bottom=max(row['bounds'][3] for row in title_plan['lines'])
    draw.rounded_rectangle((356,title_top+4,364,title_bottom-4),radius=4,fill=ACCENT)
    if subtitle:
        draw_plan(draw,text_plan(subtitle,(400,830,2392,930),48,font_path=font_path),fill=MUTED)
    return im


def node_box(node):
    required=('id','x','y','w','h','label')
    missing=[k for k in required if k not in node]
    if missing:raise LayoutError(f'Node is missing required keys: {missing}')
    x,y,w,h=(float(node[k]) for k in ('x','y','w','h'))
    return (x,y,x+w,y+h)


def anchor(box, side):
    x0,y0,x1,y1=box
    values={'left':(x0,(y0+y1)/2),'right':(x1,(y0+y1)/2),
            'top':((x0+x1)/2,y0),'bottom':((x0+x1)/2,y1)}
    if side not in values:raise LayoutError(f'Unknown explicit edge anchor: {side!r}')
    return values[side]


def line_safe(points,name, *, area=DIAGRAM_AREA, lecturer_reserve=None):
    for a,b in zip(points,points[1:]):
        length=math.hypot(b[0]-a[0],b[1]-a[1])
        if length==0:raise LayoutError(f'{name}: edge contains a zero-length segment.')
        for i in range(math.ceil(length/4)+1):
            u=i/max(1,math.ceil(length/4));x=a[0]+(b[0]-a[0])*u;y=a[1]+(b[1]-a[1])*u
            checked_box((x-8,y-8,x+8,y+8),name=name,allowed=area,lecturer_clear=True,lecturer_reserve=lecturer_reserve)


def layout_diagram(spec, *, font_path=None, profile=None, project_style=None):
    """Validate explicit geometry and return prepared plans; infer no relations.

    Spec keys: layout='explicit', title, optional subtitle, nodes, edges.
    Node and edge geometry stays inside the selected profile's content panel; the fixed
    header, white content panel, lecturer reserve and caption band are separate.
    Node: id,x,y,w,h,label, optional detail,font_size(36..52),shape,highlight.
    Edge: from,to,from_anchor,to_anchor,kind='line'|'arrow'; optional id,via,
          label,label_pos=[center_x,center_y],highlight.
    All label text and every edge must be supplied by the author.
    """
    if spec.get('layout')!='explicit':raise LayoutError('Diagram layout must be explicitly set to "explicit".')
    if not str(spec.get('title','')).strip():raise LayoutError('Diagram title is required.')
    selected = layout_settings(profile, project_style)
    area, reserve = selected['area'], selected['lecturer_clear']
    series=text_plan(SERIES,(56,46,2480,86),30,valign='top',font_path=font_path)
    title_top=int(selected['layout'].get('title_top',108))
    title=text_plan(spec['title'],(56,title_top,2480,200),DIAGRAM_TITLE_SIZE,valign='top',font_path=font_path)
    subtitle=None
    if spec.get('subtitle'):
        panel = selected['panel']
        subtitle=text_plan(spec['subtitle'],(panel[0]+40,panel[1]+12,panel[2]-40,panel[1]+52),36,valign='top',font_path=font_path)
    nodes={};boxes={}
    for node in spec.get('nodes',[]):
        b=node_box(node);name=str(node['id'])
        if name in nodes:raise LayoutError(f'Duplicate node ID: {name}')
        checked_box(b,name=f'Node {name}',allowed=area,lecturer_clear=True,lecturer_reserve=reserve)
        for old,oldbox in boxes.items():
            if overlap(b,oldbox):raise LayoutError(f'Nodes {name} and {old} overlap.')
        size=int(node.get('font_size',DIAGRAM_NODE_SIZE))
        if not DIAGRAM_FONT_MIN<=size<=DIAGRAM_FONT_MAX:raise LayoutError(f'Node {name}: font_size must be 36..52.')
        shape=node.get('shape','roundrect')
        if shape not in ('rect','roundrect','ellipse'):raise LayoutError(f'Node {name}: unsupported shape {shape}.')
        if shape=='ellipse':
            # The inscribed central rectangle keeps glyphs inside the ellipse.
            dx=(b[2]-b[0])*.16;dy=(b[3]-b[1])*.16
        else:dx=24;dy=20
        inner=(b[0]+dx,b[1]+dy,b[2]-dx,b[3]-dy)
        detail=node.get('detail','');plans=[]
        if detail:
            middle=inner[1]+(inner[3]-inner[1])*.47
            plans.append((text_plan(node['label'],(inner[0],inner[1],inner[2],middle),size,max_lines=2,line_gap=10,align='center',font_path=font_path),'label'))
            plans.append((text_plan(detail,(inner[0],middle+12,inner[2],inner[3]),36,max_lines=3,line_gap=10,align='center',font_path=font_path),'detail'))
        else:plans.append((text_plan(node['label'],inner,size,max_lines=3,line_gap=12,align='center',font_path=font_path),'label'))
        boxes[name]=b;nodes[name]=dict(node=node,box=b,plans=plans,shape=shape)
    if not nodes:raise LayoutError('At least one explicitly positioned node is required.')
    edges=[]
    used_ids=set(nodes)
    for index,edge in enumerate(spec.get('edges',[])):
        name=str(edge.get('id',f'edge-{index+1}'))
        if name in used_ids:raise LayoutError(f'Duplicate node/edge highlight ID: {name}')
        used_ids.add(name)
        if edge.get('kind') not in ('line','arrow'):raise LayoutError(f'{name}: explicitly specify kind="line" or "arrow".')
        for key in ('from','to','from_anchor','to_anchor'):
            if key not in edge:raise LayoutError(f'{name}: explicit {key} is required.')
        if edge['from'] not in nodes or edge['to'] not in nodes:raise LayoutError(f'{name}: unknown node reference.')
        via=edge.get('via',[])
        if not isinstance(via,list) or any(len(p)!=2 for p in via):raise LayoutError(f'{name}: via must be a list of [x,y] points.')
        points=[anchor(boxes[edge['from']],edge['from_anchor'])]+[tuple(map(float,p)) for p in via]+[anchor(boxes[edge['to']],edge['to_anchor'])]
        line_safe(points,name,area=area,lecturer_reserve=reserve)
        # A line must not silently pass through an unrelated node.
        for a,b in zip(points,points[1:]):
            count=max(1,math.ceil(math.dist(a,b)/4))
            for k in range(count+1):
                u=k/count;x=a[0]+(b[0]-a[0])*u;y=a[1]+(b[1]-a[1])*u
                for node_id,box in boxes.items():
                    if node_id not in (edge['from'],edge['to']) and box[0]<x<box[2] and box[1]<y<box[3]:
                        raise LayoutError(f'{name}: edge crosses unrelated node {node_id}. Specify via points.')
        label=None
        if edge.get('label'):
            if 'label_pos' not in edge:raise LayoutError(f'{name}: a label needs explicit label_pos=[x,y].')
            x,y=map(float,edge['label_pos']);face=font(36,font_path);b=face.getbbox(str(edge['label']));w=b[2]-b[0]
            box=(x-w/2-12,y-28,x+w/2+12,y+28)
            checked_box(box,name=f'{name} label',allowed=area,lecturer_clear=True,lecturer_reserve=reserve)
            if any(overlap(box,b) for b in boxes.values()):raise LayoutError(f'{name}: label overlaps a node.')
            label=text_plan(edge['label'],box,36,align='center',font_path=font_path)
        edges.append(dict(edge=edge,id=name,points=points,label=label))
    return dict(series=series,title=title,subtitle=subtitle,nodes=nodes,edges=edges,settings=selected)


def draw_arrow(draw,points,fill,width=4):
    draw.line(points,fill=fill,width=width,joint='curve')
    a,b=points[-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0]);length=18;spread=8
    back=(b[0]-length*math.cos(angle),b[1]-length*math.sin(angle))
    draw.polygon([b,(back[0]+spread*math.sin(angle),back[1]-spread*math.cos(angle)),
                  (back[0]-spread*math.sin(angle),back[1]+spread*math.cos(angle))],fill=fill)


def render_diagram(spec, *, highlight=None, font_path=None, profile=None, project_style=None):
    prepared=layout_diagram(spec,font_path=font_path,profile=profile,project_style=project_style)
    selected=set(highlight or [])
    known=set(prepared['nodes'])|{e['id'] for e in prepared['edges']}
    if selected-known:raise LayoutError(f'Unknown highlight IDs: {sorted(selected-known)}')
    im=Image.new('RGB',(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(im)
    draw.line((56,219,2480,219),fill=BORDER,width=2)
    draw.rounded_rectangle(prepared['settings']['panel'],radius=14,fill=CARD)
    draw_plan(draw,prepared['series'],fill=MUTED)
    draw_plan(draw,prepared['title'])
    if prepared['subtitle']:draw_plan(draw,prepared['subtitle'],fill=MUTED)
    for item in prepared['edges']:
        e=item['edge'];active=item['id'] in selected or bool(e.get('highlight'))
        color=ACCENT if active else MUTED
        if e['kind']=='arrow':draw_arrow(draw,item['points'],color,5 if active else 4)
        else:draw.line(item['points'],fill=color,width=5 if active else 4,joint='curve')
        if item['label']:
            draw.rectangle(item['label']['box'],fill=CARD);draw_plan(draw,item['label'],fill=color)
    for name,item in prepared['nodes'].items():
        active=name in selected or bool(item['node'].get('highlight'));b=item['box']
        options=dict(fill=HIGHLIGHT if active else CARD,outline=ACCENT if active else BORDER,width=4 if active else 3)
        if item['shape']=='ellipse':draw.ellipse(b,**options)
        elif item['shape']=='rect':draw.rectangle(b,**options)
        else:draw.rounded_rectangle(b,radius=18,**options)
        for plan,kind in item['plans']:draw_plan(draw,plan,fill=INK if kind=='label' else MUTED)
    return im


def render_subtitle(text, *, style='dark', font_path=None):
    """Transparent RGBA 2560x160 strip, placed at y=1280 by the editor."""
    if style not in ('dark','light'):raise LayoutError('Subtitle style must be dark or light.')
    im=Image.new('RGBA',(WIDTH,CAPTION_HEIGHT),(0,0,0,0))
    if not str(text).strip():return im
    plan=text_plan(text,(80,14,WIDTH-80,CAPTION_HEIGHT-14),SUBTITLE_SIZE,max_lines=2,line_gap=14,align='center',font_path=font_path)
    draw_plan(ImageDraw.Draw(im),plan,fill=INK if style=='dark' else '#FFFFFF',
        stroke_width=0 if style=='dark' else 3,stroke_fill=None if style=='dark' else (0,0,0,210))
    return im


def render_transition(frame_index, title, *, font_path=None):
    """Full opaque RGB frame, local frame index 0..77, with a soft title fade.

    There is no transparent or lecturer-safe region anywhere. The blank frame
    endpoints remain opaque BG, so an underlying portrait can never show through.
    Do not add a portrait layer after this render; its external guards are editor-owned.
    """
    n=int(frame_index)
    if n!=frame_index or not 0<=n<TRANSITION_FRAMES:raise LayoutError('Transition frame must be an integer from 0 through 77.')
    if not str(title).strip():raise LayoutError('Transition title must not be empty.')
    im=Image.new('RGB',(WIDTH,HEIGHT),BG)
    face=font(TITLE_SIZE,font_path);bbox=face.getbbox(title);text_width=bbox[2]-bbox[0]
    if text_width>2050 or '\n' in title:raise LayoutError('Transition title must fit one line at 120px within 2050px.')
    box=((WIDTH-text_width)/2+23,520,(WIDTH+text_width)/2+23,760)
    plan=text_plan(title,box,TITLE_SIZE,align='center',font_path=font_path)
    t=n/FPS;last=(TRANSITION_FRAMES-1)/FPS
    alpha=smooth(t/.35)*(1-smooth((t-.95)/(last-.95)))
    if alpha<=0:return im
    overlay=Image.new('RGBA',(WIDTH,HEIGHT),(0,0,0,0));draw=ImageDraw.Draw(overlay)
    draw_plan(draw,plan)
    x=box[0]-46;draw.rounded_rectangle((round(x),608,round(x+8),672),radius=4,fill=ACCENT)
    overlay.putalpha(overlay.getchannel('A').point([round(v*alpha) for v in range(256)]))
    return Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')


def save_png(image,path):
    path=Path(path)
    if path.suffix.lower()!='.png':raise LayoutError('Output must use a .png extension.')
    path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
    return path.resolve()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--cover',action='store_true')
    modes.add_argument('--diagram',type=Path,metavar='SPEC.json')
    modes.add_argument('--transition',action='store_true')
    modes.add_argument('--caption',metavar='TEXT',help='Transparent 2560x160 subtitle strip; --subtitle supplies the cover secondary text.')
    parser.add_argument('--title',default='')
    parser.add_argument('--subtitle',default='')
    parser.add_argument('--speaker',default=STYLE.get('speaker','讲师'))
    parser.add_argument('--series',default=SERIES)
    parser.add_argument('--label',default='')
    parser.add_argument('--font',default=None,help='Explicit font path; otherwise COURSE_BRAND_FONT or discovered system CJK font.')
    styles=parser.add_mutually_exclusive_group()
    styles.add_argument('--profile',choices=['theory','practical','mixed'],help='Diagram layout: theory uses a left lecturer; practical/mixed use a right lecturer.')
    styles.add_argument('--project-style',type=Path,help='Diagram layout from project init\'s resolved project-style.json, including per-lesson overrides.')
    parser.add_argument('--highlight',default='',help='Comma-separated explicit diagram node/edge IDs; geometry never changes.')
    parser.add_argument('--caption-style',choices=['dark','light'],default='dark')
    parser.add_argument('--preview',action='store_true',help='For transition only: write frames 0,24,39,77.')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--out-dir',type=Path)
    args=parser.parse_args(argv)
    try:
        if (args.profile or args.project_style) and not args.diagram:
            parser.error('--profile and --project-style apply only to diagram mode.')
        if args.transition:
            if not args.out_dir:parser.error('--transition requires --out-dir')
            indices=[0,24,39,77] if args.preview else range(TRANSITION_FRAMES)
            results=[str(save_png(render_transition(n,args.title,font_path=args.font),args.out_dir/f'frame-{n:03d}.png')) for n in indices]
            print(json.dumps(dict(mode='transition',fps=FPS,duration=TRANSITION_SECONDS,frames=len(results),opaque=True,files=results),ensure_ascii=False))
            return 0
        if args.preview:parser.error('--preview is supported only with --transition')
        if not args.out:parser.error('This mode requires --out output.png')
        if args.cover:im=render_cover(args.title,args.subtitle,args.speaker,series=args.series,label=args.label,font_path=args.font)
        elif args.diagram:
            spec=json.loads(args.diagram.read_text(encoding='utf-8-sig'))
            im=render_diagram(spec,highlight=[s.strip() for s in args.highlight.split(',') if s.strip()],font_path=args.font,profile=args.profile,project_style=args.project_style)
        else:im=render_subtitle(args.caption,style=args.caption_style,font_path=args.font)
        out=save_png(im,args.out)
        print(json.dumps(dict(path=str(out),width=im.width,height=im.height,mode=im.mode),ensure_ascii=False));return 0
    except (LayoutError,KeyError,TypeError,OSError,json.JSONDecodeError) as exc:
        parser.error(str(exc))

if __name__=='__main__':raise SystemExit(main())
