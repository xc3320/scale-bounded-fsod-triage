# -*- coding: utf-8 -*-
"""Paper 4 Figure 1 (v3): compact, high-contrast journal style.
Fixes over v2 per user review: larger minimum font sizes, tighter vertical
rhythm, darker secondary text (no faint grays), stronger accent colors.
Pillow only. All numbers verbatim from 成稿事实清单.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter

S = 2                                  # supersample
LW, LH = 3000, 2000                    # logical canvas (was 2400 -> tighter)
W, H = LW * S, LH * S
FR, FB = "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"

def f(sz, bold=False):
    return ImageFont.truetype(FB if bold else FR, sz * S)

img = Image.new("RGBA", (W, H), "#F7F9FCFF")
d = ImageDraw.Draw(img)

# palette (v3: darker secondaries, stronger accents)
NAVY   = "#12263F"
BLUE   = "#1F5C8B"
TEAL   = "#0E7C7B"
TEALD  = "#0B5F5E"   # dark teal for body text
AMBER  = "#9A6412"   # darker amber (was B5761A)
AMBERD = "#8F5A0E"
RED    = "#A83232"
GREEN  = "#0B6B4F"   # darker green (was 0E7C5A)
INK    = "#1B2839"   # darker body ink
MUTED  = "#48566B"   # darker secondary (was 5C6B7F)
CARD1  = "#FFFFFF"
LINE1  = "#D8E2EC"

def rr(box, fill, outline=None, width=2, radius=20):
    d.rounded_rectangle(tuple(int(v*S) for v in box), radius=radius*S,
                        fill=fill, outline=outline, width=width*S if outline else 1)

def shadow(box, radius=20, blur=18, alpha=26):
    x0,y0,x1,y1 = box
    lay = Image.new("RGBA", (W,H), (0,0,0,0))
    dd = ImageDraw.Draw(lay)
    dd.rounded_rectangle((int((x0+6)*S),int((y0+10)*S),int((x1+6)*S),int((y1+10)*S)),
                         radius=radius*S, fill=(18,38,63,alpha))
    lay = lay.filter(ImageFilter.GaussianBlur(blur*S))
    img.alpha_composite(lay)

def tx(xy, s, font, fill, anchor="la"):
    d.text((int(xy[0]*S), int(xy[1]*S)), s, font=font, fill=fill, anchor=anchor)

def rule(x1,y1,x2,y2,color=LINE1,w=2):
    d.line([(x1*S,y1*S),(x2*S,y2*S)], fill=color, width=w*S)

def dash(x1,x2,y,color=MUTED):
    x=x1
    while x<x2:
        d.line([(x*S,y*S),(min(x+22,x2)*S,y*S)], fill=color, width=2*S); x+=34

import math
def arrow(p1,p2,color="#7E93AA",width=5,head=22):
    d.line([(p1[0]*S,p1[1]*S),(p2[0]*S,p2[1]*S)], fill=color, width=width*S)
    a=math.atan2(p2[1]-p1[1],p2[0]-p1[0])
    for off in (2.55,-2.55):
        q=(p2[0]+head*math.cos(a+off), p2[1]+head*math.sin(a+off))
        d.line([(p2[0]*S,p2[1]*S),(q[0]*S,q[1]*S)], fill=color, width=width*S)

# ============ HEADER (compact) ============
tx((LW//2, 64), "From Diagnosis to Method", f(56,True), NAVY, anchor="ma")
tx((LW//2, 128), "A scale-bounded, mechanism-driven triage for training-free few-shot detection",
   f(34), MUTED, anchor="ma")
rule(1180,176,1820,176,color="#C9D6E4",w=3)

# ============ STAGE 1 : DIAGNOSE (120,206 - 1450,960) ============
P1=(120,206,1450,960)
rr(P1, CARD1, LINE1, width=2); shadow(P1)
rr((120,206,1450,290), BLUE, None, radius=20)
d.rectangle((120*S,258*S,1450*S,290*S), fill=BLUE)
d.ellipse(((178-24)*S,(248-24)*S,(178+24)*S,(248+24)*S), fill="#FFFFFF")
d.text((178*S,248*S),"1",font=f(30,True),fill=BLUE,anchor="mm")
tx((222,248),"DIAGNOSE  |  scale-bounded decomposition", f(38,True), "#FFFFFF", anchor="lm")

tx((172,318),"P(bg | s, g) = F(s) + G(g, s)", f(46,True), NAVY)
tx((172,382),"F(s): scale-general effect      G: few-shot penalty", f(34), MUTED)
rule(172,430,1398,430,LINE1)

tx((186,458),"scale", f(33,True), BLUE)
tx((600,458),"LOSO MAE", f(33,True), BLUE)
tx((960,458),"novel n", f(33,True), BLUE)
tx((1180,458),"verdict", f(33,True), BLUE)
rows=[("small","25.46 pp","12-44","INDETERMINATE",RED),
      ("medium","2.74 pp","217-392","predictable",GREEN),
      ("large","8.51 pp","827-1238","predictable",GREEN)]
ys=[508,580,652]
for (a,b,c,e,col),y in zip(rows,ys):
    tx((186,y),a,f(35,True),INK); tx((600,y),b,f(35),INK)
    tx((960,y),c,f(34),INK)
    if e=="INDETERMINATE":
        rr((1172,y-12,1410,y+42),"#F5DBDB",None,radius=10)
        tx((1291,y+15),e,f(30,True),col,anchor="mm")
    else:
        tx((1180,y),e,f(32,True),col)
dash(172,1398,622)
tx((172,712),"medium + large merged: MAE 5.62 pp < 8 pp", f(36,True), GREEN)
tx((172,776),"small: n <= 44, 95% CI half-width 11-24 pp", f(32), MUTED)
tx((172,838),"dashed rule = scale predictability boundary", f(31), MUTED)

# ============ STAGE 2 : COLLAPSE (1550,206 - 2880,960) ============
P2=(1550,206,2880,960)
rr(P2, CARD1, LINE1, width=2); shadow(P2)
rr((1550,206,2880,290), AMBER, None, radius=20)
d.rectangle((1550*S,258*S,2880*S,290*S), fill=AMBER)
d.ellipse(((1610-24)*S,(248-24)*S,(1610+24)*S,(248+24)*S), fill="#FFFFFF")
d.text((1610*S,248*S),"2",font=f(30,True),fill="#FFFFFF",anchor="mm")
tx((1654,248),"EXPLAIN  |  why uniform repair fails", f(38,True), "#FFFFFF", anchor="lm")

tx((1604,318),"Uniform training-time foreground margin (SAFM-style)", f(33), MUTED)
cmps=[("GT correct","0.247 -> 0.634","mechanism moves",GREEN),
      ("BG rate","95.5% -> 63.6%","",MUTED),
      ("novel mAP","25.25 -> 14.96","COLLAPSES",RED)]
cy=[392,478,564]
for (name,val,note,col),y in zip(cmps,cy):
    tx((1610,y),name,f(34,True),INK)
    tx((1870,y+2),val,f(40,True),col)
    if note: tx((2530,y+8),note,f(31,True),col)
rule(1604,634,2826,634,LINE1)
tx((1604,662),"per-class divergence at all three margin strengths", f(33), MUTED)
rr((1604,716,2826,826),"#F9EDD8","#D9B271",width=2,radius=14)
tx((2215,771),"=> one-size-fits-all fix overfits scarce positives", f(36,True), AMBERD, anchor="mm")
tx((1604,876),"negative control  ->  motivates triage", f(32), MUTED)

# ============ arrows (short) ============
arrow((790,960),(1010,1058))
arrow((2215,960),(1995,1058))

# ============ STAGE 3 : TRIAGE (120,1058 - 2880,1700) ============
P3=(120,1058,2880,1700)
rr(P3, CARD1, LINE1, width=2); shadow(P3)
rr((120,1058,2880,1142), TEAL, None, radius=20)
d.rectangle((120*S,1110*S,2880*S,1142*S), fill=TEAL)
d.ellipse(((178-24)*S,(1100-24)*S,(178+24)*S,(1100+24)*S), fill="#FFFFFF")
d.text((178*S,1100*S),"3",font=f(30,True),fill=TEAL,anchor="mm")
tx((222,1100),"REPAIR  |  scale-conditioned semantic triage (training-free)", f(38,True), "#FFFFFF", anchor="lm")

rr((180,1188,1380,1500),"#EFF7F3","#A8D8C6",width=2,radius=16)
tx((780,1226),"SMALL proposals", f(37,True), TEALD, anchor="ma")
tx((780,1290),"gate weight = 0  (ABSTAIN)", f(35,True), INK, anchor="ma")
tx((780,1352),"indeterminate G; oracle ceiling 30.10", f(31), MUTED, anchor="ma")
tx((780,1408),"dual bottleneck -> not repairable here", f(31), MUTED, anchor="ma")

rr((1620,1188,2820,1500),"#EFF7F3","#A8D8C6",width=2,radius=16)
tx((2220,1226),"MEDIUM + LARGE proposals", f(37,True), TEALD, anchor="ma")
tx((2220,1290),"CLIP gating ON (ViT-B/32)", f(35,True), INK, anchor="ma")
tx((2220,1352),"predictable G (5.62 pp); gain supported", f(31), MUTED, anchor="ma")
tx((2220,1408),"per-proposal scale gate, no retraining", f(31), MUTED, anchor="ma")

rr((420,1544,2580,1624),"#DDF0E7",None,radius=14)
tx((1500,1584),"hard gate +9.57 pp   |   ungated +9.47 pp   |   soft gate +6.35 pp (FAIL)",
   f(36,True), TEALD, anchor="mm")
tx((1500,1656),"mean novel-mAP gain over 3 VOC splits; base mAP never degraded", f(30), MUTED, anchor="ma")

# ============ ORACLE BAR (120,1740 - 2880,1936) ============
arrow((1500,1700),(1500,1768))
rr((120,1768,2880,1936),"#EDF1F7","#AFC1D6",width=2,radius=18)
tx((1500,1800),"ORACLE CEILING  (perfect classifier, localization unchanged)", f(34,True), NAVY, anchor="ma")
tx((1500,1846),"small 30.10     medium 44.26     large 64.90", f(40,True), NAVY, anchor="ma")
tx((1500,1896),"small objects: classification + localization bottleneck -> beyond classification-side repair",
   f(30), MUTED, anchor="ma")

out=img.resize((LW,LH), Image.LANCZOS).convert("RGB")
OUT="E:/WorkBuddy/spaces/2026_11_18_08-47-15/论文4_Figure1_mechanism.png"
out.save(OUT,dpi=(300,300))
print("saved v3:",OUT,"size:",out.size)
