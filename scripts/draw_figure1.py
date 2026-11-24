# -*- coding: utf-8 -*-
"""Paper 4 Figure 1 (v2, premium journal style): mechanism-to-method pipeline.
Refined palette, generous whitespace, numbered stage badges, icon-like markers,
subtle card shadows and a clean visual hierarchy. Pillow only (no matplotlib).
All numbers come verbatim from 成稿事实清单 / EN master v0.2.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter

S = 2                                  # supersample
LW, LH = 3000, 2400                    # logical canvas
W, H = LW * S, LH * S
FR, FB = "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"

def f(sz, bold=False):
    return ImageFont.truetype(FB if bold else FR, sz * S)

img = Image.new("RGBA", (W, H), "#F7F9FCFF")
d = ImageDraw.Draw(img)

# premium palette
NAVY   = "#12263F"   # deep navy (headings)
BLUE   = "#1F5C8B"   # steel blue (stage 1)
TEAL   = "#0E7C7B"   # teal (stage 3)
AMBER  = "#B5761A"   # amber (stage 2 / warning)
RED    = "#A83232"   # muted red (collapse)
GREEN  = "#0E7C5A"   # green (pass)
INK    = "#223247"   # body ink
MUTED  = "#5C6B7F"   # secondary
FAINT  = "#8C9AAB"   # tertiary
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

def badge(cx, cy, n, color):
    r = 26
    d.ellipse(((cx-r)*S,(cy-r)*S,(cx+r)*S,(cy+r)*S), fill=color)
    d.text((cx*S, cy*S), str(n), font=f(30,True), fill="#FFFFFF", anchor="mm")

def rule(x1,y1,x2,y2,color=LINE1,w=2):
    d.line([(x1*S,y1*S),(x2*S,y2*S)], fill=color, width=w*S)

def dash(x1,x2,y,color=FAINT):
    x=x1
    while x<x2:
        d.line([(x*S,y*S),(min(x+22,x2)*S,y*S)], fill=color, width=2*S); x+=34

import math
def arrow(p1,p2,color=MUTED,width=5,head=22):
    d.line([(p1[0]*S,p1[1]*S),(p2[0]*S,p2[1]*S)], fill=color, width=width*S)
    a=math.atan2(p2[1]-p1[1],p2[0]-p1[0])
    for off in (2.55,-2.55):
        q=(p2[0]+head*math.cos(a+off), p2[1]+head*math.sin(a+off))
        d.line([(p2[0]*S,p2[1]*S),(q[0]*S,q[1]*S)], fill=color, width=width*S)

# ============ HEADER ============
tx((LW//2, 78), "From Diagnosis to Method", f(58,True), NAVY, anchor="ma")
tx((LW//2, 148), "A scale-bounded, mechanism-driven triage for training-free few-shot detection",
   f(34), MUTED, anchor="ma")
rule(1180,196,1820,196,color="#C9D6E4",w=3)

# ============ STAGE 1 : DIAGNOSE ============
rr((120,232,1450,1140), CARD1, LINE1, width=2); shadow((120,232,1450,1140))
rr((120,232,1450,318), BLUE, None, radius=20)   # top band
# square off band bottom
d.rectangle((120*S,285*S,1450*S,318*S), fill=BLUE)
badge(178,275,"1","#FFFFFF")
# badge number color override -> navy text on white circle looks better; redraw
d.ellipse(((178-26)*S,(275-26)*S,(178+26)*S,(275+26)*S), fill="#FFFFFF")
d.text((178*S,275*S),"1",font=f(30,True),fill=BLUE,anchor="mm")
tx((224,275),"DIAGNOSE  |  scale-bounded decomposition", f(38,True), "#FFFFFF")

tx((172,352),"P(bg | s, g) = F(s) + G(g, s)", f(44,True), NAVY)
tx((172,414),"F(s): scale-general effect     G: few-shot penalty", f(31), MUTED)
rule(172,458,1398,458,LINE1)

tx((186,492),"scale", f(30,True), BLUE)
tx((600,492),"LOSO MAE", f(30,True), BLUE)
tx((960,492),"novel n", f(30,True), BLUE)
tx((1180,492),"verdict", f(30,True), BLUE)
rows=[("small","25.46 pp","12-44","INDETERMINATE",RED),
      ("medium","2.74 pp","217-392","predictable",GREEN),
      ("large","8.51 pp","827-1238","predictable",GREEN)]
ys=[556,660,748]
for (a,b,c,e,col),y in zip(rows,ys):
    tx((186,y),a,f(33,True),INK); tx((600,y),b,f(33),INK)
    tx((960,y),c,f(32),INK)
    if e=="INDETERMINATE":
        rr((1172,y-14,1400,y+40),"#FBEAEA",None,radius=10)   # chip BEFORE text
        tx((1286,y+14),e,f(27,True),col,anchor="ma")          # centered in chip
    else:
        tx((1180,y),e,f(30,True),col)
dash(172,1398,626)
tx((172,838),"medium + large merged: MAE 5.62 pp < 8 pp", f(33,True), GREEN)
tx((172,906),"small: n <= 44, 95% CI half-width 11-24 pp", f(30), MUTED)
tx((172,985),"dashed rule = scale predictability boundary", f(28), FAINT)

# ============ STAGE 2 : COLLAPSE ============
rr((1550,232,2880,1140), CARD1, LINE1, width=2); shadow((1550,232,2880,1140))
rr((1550,232,2880,318), AMBER, None, radius=20)
d.rectangle((1550*S,285*S,2880*S,318*S), fill=AMBER)
d.ellipse(((1610-26)*S,(275-26)*S,(1610+26)*S,(275+26)*S), fill="#FFFFFF")
d.text((1610*S,275*S),"2",font=f(30,True),fill=AMBER,anchor="mm")
tx((1656,275),"EXPLAIN  |  why uniform repair fails", f(38,True), "#FFFFFF")

tx((1604,352),"Uniform training-time foreground margin (SAFM-style)", f(30), MUTED)
cmps=[("GT correct","0.247 -> 0.634","mechanism moves",GREEN),
      ("BG rate","95.5% -> 63.6%","",MUTED),
      ("novel mAP","25.25 -> 14.96","COLLAPSES",RED)]
cy=[430,524,620]
for (name,val,note,col),y in zip(cmps,cy):
    tx((1610,y),name,f(32,True),INK)
    tx((1870,y),val,f(38,True),col)
    if note: tx((2510,y+3),note,f(28,True),col)
rule(1604,706,2826,706,LINE1)
tx((1604,742),"per-class divergence at all three margin strengths", f(30), MUTED)
# warning callout
rr((1604,800,2826,906),"#FBF3E4","#E7C98F",width=2,radius=14)
tx((1640,853),"=> one-size-fits-all fix overfits scarce positives", f(34,True), AMBER)
tx((1604,972),"negative control  ->  motivates triage", f(30), MUTED)

# ============ arrows ============
arrow((790,1140),(1010,1312),color="#9FB2C6")
arrow((2215,1140),(1995,1312),color="#9FB2C6")

# ============ STAGE 3 : TRIAGE ============
rr((120,1312,2880,2076), CARD1, LINE1, width=2); shadow((120,1312,2880,2076))
rr((120,1312,2880,1398), TEAL, None, radius=20)
d.rectangle((120*S,1365*S,2880*S,1398*S), fill=TEAL)
d.ellipse(((178-26)*S,(1355-26)*S,(178+26)*S,(1355+26)*S), fill="#FFFFFF")
d.text((178*S,1355*S),"3",font=f(30,True),fill=TEAL,anchor="mm")
tx((224,1355),"REPAIR  |  scale-conditioned semantic triage (training-free)", f(37,True), "#FFFFFF")

# two sub-cards
rr((180,1444,1380,1808),"#F4FAF7","#BFE3D5",width=2,radius=16)
tx((780,1486),"SMALL proposals", f(37,True), TEAL, anchor="ma")
tx((780,1558),"gate weight = 0  (ABSTAIN)", f(33,True), INK, anchor="ma")
tx((780,1628),"indeterminate G; oracle ceiling 30.10", f(29), MUTED, anchor="ma")
tx((780,1692),"dual bottleneck -> not repairable here", f(29), MUTED, anchor="ma")

rr((1620,1444,2820,1808),"#F4FAF7","#BFE3D5",width=2,radius=16)
tx((2220,1486),"MEDIUM + LARGE proposals", f(37,True), TEAL, anchor="ma")
tx((2220,1558),"CLIP gating ON (ViT-B/32)", f(33,True), INK, anchor="ma")
tx((2220,1628),"predictable G (5.62 pp); gain supported", f(29), MUTED, anchor="ma")
tx((2220,1692),"per-proposal scale gate, no retraining", f(29), MUTED, anchor="ma")

rr((420,1870,2580,1950),"#EAF6F0",None,radius=14)
tx((1500,1910),"hard gate +9.57 pp   |   ungated +9.47 pp   |   soft gate +6.35 pp (FAIL)",
   f(34,True), TEAL, anchor="mm")
tx((1500,2006),"mean novel-mAP gain over 3 VOC splits; base mAP never degraded", f(28), MUTED, anchor="ma")

# ============ ORACLE BAR ============
arrow((1500,2076),(1500,2152),color="#9FB2C6")
rr((120,2152,2880,2370),"#EEF2F8","#B9C8DC",width=2,radius=18)
tx((1500,2192),"ORACLE CEILING  (perfect classifier, localization unchanged)", f(34,True), NAVY, anchor="ma")
tx((1500,2256),"small 30.10     medium 44.26     large 64.90", f(38,True), NAVY, anchor="ma")
tx((1500,2321),"small objects: classification + localization bottleneck -> beyond classification-side repair",
   f(28), MUTED, anchor="ma")

out=img.resize((LW,LH), Image.LANCZOS).convert("RGB")
OUT="E:/WorkBuddy/spaces/2026_11_18_08-47-15/论文4_Figure1_mechanism.png"
out.save(OUT,dpi=(300,300))
print("saved v2:",OUT,"size:",out.size)
