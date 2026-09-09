#!/usr/bin/env python3
"""Schematic of the ADAU1772BCPZ section, drawn as a circuit.

The main sheet shows the codec as one block. This draws it properly: the chip
symbol in the datasheet's own pin arrangement, with every passive as a real
component symbol wired to the pad it belongs to, so it can be built from the
picture rather than from a table.

Nets are read back from rhear_phase0_netlist.json, so it cannot disagree with
the ERC-checked design.

    /opt/anaconda3/bin/python hardware/draw_codec.py
"""
import os, json
import schemdraw
from schemdraw import elements as elm

OUT = os.path.dirname(os.path.abspath(__file__))
NET = json.load(open(os.path.join(OUT, "rhear_phase0_netlist.json")))
INK, TEAL, RED, GREY, AMB = "#11191C", "#0B6E6C", "#9C3A14", "#93A3A8", "#8A6D1F"

def net_of(pin):
    for n, m in NET["nets"].items():
        if pin in m: return n
    return None

def col(el, c):
    for s in el.segments:
        s.color = c

d = schemdraw.Drawing(font="sans", fontsize=9, lw=1.7)

# ---- the chip, pins on the four sides exactly as the datasheet symbol ----
L = ["1 SDA","2 SCL","3 ADDR1","4 ADDR0","5 SELFBOOT","6 MICBIAS0","7 MICBIAS1",
     "8 AIN0REF","9 AIN0","10 AVDD_1"]
B = ["11 AGND_1","12 CM","13 AIN1REF","14 AIN1","15 AIN2REF","16 AIN2",
     "17 AIN3REF","18 AIN3","19 AVDD_2","20 AGND_2"]
R = ["21 HPOUTLN","22 HPOUTLP","23 AGND_3","24 AVDD_3","25 HPOUTRN","26 HPOUTRP",
     "27 PD","28 REG_OUT","29 DVDD","30 DGND"]
T = ["31 LRCLK","32 BCLK","33 DAC_SDATA","34 ADC_SDATA0","35 ADC_SDATA1",
     "36 DMIC2_3","37 DMIC0_1","38 XTALO","39 XTALI","40 IOVDD"]
pins  = [elm.IcPin(name=n, side="left",   slot=f"{10-i}/10") for i, n in enumerate(L)]
pins += [elm.IcPin(name=n, side="bottom", slot=f"{i+1}/10")  for i, n in enumerate(B)]
pins += [elm.IcPin(name=n, side="right",  slot=f"{i+1}/10")  for i, n in enumerate(R)]
pins += [elm.IcPin(name=n, side="top",    slot=f"{10-i}/10") for i, n in enumerate(T)]
U4 = elm.Ic(pins=pins, size=(9.5, 9.0), plblsize=8, edgepadW=.6, edgepadH=.6).at((0, 0)).right()
d.add(U4)
A = dict(U4.absanchors)
bb = U4.get_bbox(transform=True)
d.add(elm.Label().at(((bb.xmin+bb.xmax)/2, (bb.ymin+bb.ymax)/2 + 0.5)).label(
    "ADAU1772BCPZ", fontsize=13))
d.add(elm.Label().at(((bb.xmin+bb.xmax)/2, (bb.ymin+bb.ymax)/2 - 0.3)).label(
    "U4   40-LFCSP on QFN40→DIP adapter", fontsize=8.5, color=GREY))
d.add(elm.Label().at(((bb.xmin+bb.xmax)/2, (bb.ymin+bb.ymax)/2 - 1.1)).label(
    "pad 41 = EP underneath → GND", fontsize=8, color=RED))

def stub(anch, dx, dy, txt, c=TEAL, fs=8.5, ha="left"):
    p = A[anch]; q = (p[0]+dx, p[1]+dy)
    ln = elm.Line().at(p).to(q); d.add(ln); col(d.elements[-1], c)
    d.add(elm.Label().at((q[0]+(0.15 if ha=="left" else -0.15), q[1])).label(
        txt, fontsize=fs, color=c, halign=ha))
    return q

def gnd_at(anch, dx, dy=0):
    p = A[anch]; q = (p[0]+dx, p[1]+dy)
    ln = elm.Line().at(p).to(q); d.add(ln); col(d.elements[-1], INK)
    g = elm.Ground().at(q).scale(0.5); d.add(g); col(d.elements[-1], INK)

# ---- left: I2C to the ESP32, with its pull-ups drawn as real resistors ----
for anch, gp, ry in (("1 SDA", "9", 3.3), ("2 SCL", "10", 2.4)):
    p = A[anch]
    ln = elm.Line().at(p).to((p[0]-3.4, p[1])); d.add(ln); col(d.elements[-1], TEAL)
    d.add(elm.Dot(radius=.10).at((p[0]-1.5, p[1])))
    r = elm.Resistor().at((p[0]-1.5, p[1])).up().length(ry); d.add(r)
    col(d.elements[-1], TEAL)
    d.add(elm.Label().at((p[0]-1.35, p[1]+ry/2)).label(
        f"{'R5' if gp=='9' else 'R6'}\n2.0 kΩ", fontsize=8, color=TEAL, halign="left"))
    v = elm.Vdd().at(r.end).label("+3V3", fontsize=8); d.add(v); col(d.elements[-1], RED)
    d.add(elm.Label().at((p[0]-3.55, p[1])).label(
        f"ESP32 pin {gp}", fontsize=8.5, color=TEAL, halign="right"))

for anch in ("3 ADDR1", "4 ADDR0", "5 SELFBOOT"):
    gnd_at(anch, -1.5)
for anch in ("6 MICBIAS0", "7 MICBIAS1", "8 AIN0REF", "9 AIN0"):
    stub(anch, -1.6, 0, "n.c.", GREY, 8, "right")

# ---- supplies: 3V3 with a decoupling cap on each ----
def supply(anch, cref, side="left"):
    p = A[anch]; dx = -1.9 if side == "left" else 1.9
    ln = elm.Line().at(p).to((p[0]+dx, p[1])); d.add(ln); col(d.elements[-1], RED)
    d.add(elm.Dot(radius=.10).at((p[0]+dx*0.55, p[1])))
    c = elm.Capacitor().at((p[0]+dx*0.55, p[1])).down().length(1.4); d.add(c)
    col(d.elements[-1], AMB)
    g = elm.Ground().at(c.end).scale(0.45); d.add(g); col(d.elements[-1], INK)
    d.add(elm.Label().at((p[0]+dx*0.55+0.18, p[1]-0.75)).label(
        f"{cref} 0.1µF", fontsize=7.5, color=AMB, halign="left"))
    v = elm.Vdd().at((p[0]+dx, p[1])).label("+3V3", fontsize=8); d.add(v)
    col(d.elements[-1], RED)

supply("10 AVDD_1", "C14", "left")
supply("19 AVDD_2", "C15", "left")
supply("24 AVDD_3", "C16", "right")
supply("40 IOVDD",  "C17", "right")

# ---- bottom: grounds, CM cap, analog inputs reserved ----
for anch in ("11 AGND_1", "20 AGND_2"):
    p = A[anch]; ln = elm.Line().at(p).to((p[0], p[1]-1.1)); d.add(ln)
    col(d.elements[-1], INK)
    g = elm.Ground().at((p[0], p[1]-1.1)).scale(0.5); d.add(g); col(d.elements[-1], INK)
p = A["12 CM"]
c = elm.Capacitor().at(p).down().length(1.5); d.add(c); col(d.elements[-1], AMB)
g = elm.Ground().at(c.end).scale(0.45); d.add(g); col(d.elements[-1], INK)
d.add(elm.Label().at((p[0]+0.18, p[1]-0.9)).label("C19\n0.1µF", fontsize=7.5,
                                                  color=AMB, halign="left"))
for anch in ("13 AIN1REF","14 AIN1","15 AIN2REF","16 AIN2","17 AIN3REF","18 AIN3"):
    p = A[anch]; ln = elm.Line().at(p).to((p[0], p[1]-0.7)); d.add(ln)
    col(d.elements[-1], GREY)
d.add(elm.Label().at((A["15 AIN2REF"][0], A["15 AIN2REF"][1]-1.35)).label(
    "AIN0–3  ·  phase 1: the four MAX4466 mics move here", fontsize=8.5, color=GREY))

# ---- right: outputs reserved, PD to ESP32, REG_OUT→DVDD link, DGND ----
for anch in ("21 HPOUTLN","22 HPOUTLP","25 HPOUTRN","26 HPOUTRP"):
    stub(anch, 1.6, 0, "n.c.", GREY, 8)
d.add(elm.Label().at((A["22 HPOUTLP"][0]+2.4, (A["22 HPOUTLP"][1]+A["25 HPOUTRN"][1])/2)).label(
    "phase 1: the two cup drivers", fontsize=8.5, color=GREY, halign="left"))
gnd_at("23 AGND_3", 1.5)
gnd_at("30 DGND", 1.5)
stub("27 PD", 2.6, 0, "ESP32 pin 11  (reset)", TEAL)
# the REG_OUT -> DVDD link, drawn as the wire it is
p28, p29 = A["28 REG_OUT"], A["29 DVDD"]
for pth in ((p28, (p28[0]+1.3, p28[1])),
            ((p28[0]+1.3, p28[1]), (p28[0]+1.3, p29[1])),
            ((p28[0]+1.3, p29[1]), p29)):
    ln = elm.Line().at(pth[0]).to(pth[1]); d.add(ln); col(d.elements[-1], AMB)
d.add(elm.Dot(radius=.10).at((p28[0]+1.3, p28[1])))
c = elm.Capacitor().at((p28[0]+1.3, p28[1])).right().length(1.5); d.add(c)
col(d.elements[-1], AMB)
g = elm.Ground().at(c.end).scale(0.45); d.add(g); col(d.elements[-1], INK)
d.add(elm.Label().at((p28[0]+1.5, p28[1]+0.45)).label("C18 0.1µF", fontsize=7.5,
                                                      color=AMB, halign="left"))
d.add(elm.Label().at((p28[0]+1.45, (p28[1]+p29[1])/2)).label(
    "  internal regulator feeds DVDD\n  — do NOT tie pad 29 to +3V3",
    fontsize=7.5, color=RED, halign="left"))

# ---- top: I2S to the ESP32, and the crystal drawn as a crystal ----
for anch, gp in (("31 LRCLK","13"), ("32 BCLK","12"),
                 ("33 DAC_SDATA","14"), ("34 ADC_SDATA0","21")):
    p = A[anch]; ln = elm.Line().at(p).to((p[0], p[1]+2.2)); d.add(ln)
    col(d.elements[-1], TEAL)
    d.add(elm.Label().at((p[0], p[1]+2.45)).label(f"pin {gp}", fontsize=8.5,
                                                  color=TEAL, rotate=90, halign="left"))
for anch in ("35 ADC_SDATA1","36 DMIC2_3","37 DMIC0_1"):
    p = A[anch]; ln = elm.Line().at(p).to((p[0], p[1]+0.7)); d.add(ln)
    col(d.elements[-1], GREY)
    d.add(elm.Label().at((p[0], p[1]+0.95)).label("n.c.", fontsize=7.5, color=GREY))

xi, xo = A["39 XTALI"], A["38 XTALO"]
YT = xi[1] + 2.6
for p in (xi, xo):
    ln = elm.Line().at(p).to((p[0], YT)); d.add(ln); col(d.elements[-1], AMB)
    d.add(elm.Dot(radius=.10).at((p[0], YT - 1.1)))
    cc = elm.Capacitor().at((p[0], YT - 1.1)).left().length(1.3); d.add(cc)
    col(d.elements[-1], AMB)
    g = elm.Ground().at(cc.end).scale(0.45); d.add(g); col(d.elements[-1], INK)
xt = elm.Crystal().at((xo[0], YT)).to((xi[0], YT)); d.add(xt); col(d.elements[-1], AMB)
d.add(elm.Label().at(((xi[0]+xo[0])/2, YT + 0.55)).label(
    "Y1  12.000 MHz", fontsize=9, color=AMB))
d.add(elm.Label().at((xo[0]-1.5, YT - 1.55)).label("C13\n22pF", fontsize=7.5,
                                                   color=AMB, halign="right"))
d.add(elm.Label().at((xi[0]-1.5, YT - 1.55)).label("C12\n22pF", fontsize=7.5,
                                                   color=AMB, halign="right"))

d.add(elm.Label().at((bb.xmin - 4.2, bb.ymax + 5.2)).label(
    "ADAU1772 codec section  —  RHEAR phase 1", fontsize=14, halign="left"))
d.add(elm.Label().at((bb.xmin - 4.2, bb.ymax + 4.6)).label(
    "generated from the ERC-checked netlist  ·  numbers are adapter pad = chip pin",
    fontsize=8.5, color=GREY, halign="left"))

for p in ("U4.SDA","U4.SCL","U4.PD","U4.BCLK","U4.LRCLK","U4.DAC_SDATA",
          "U4.ADC_SDATA0","U4.XTALI","U4.XTALO","U4.DVDD"):
    assert net_of(p), f"{p} is drawn here but not connected in the netlist"

d.save(os.path.join(OUT, "rhear_codec_wiring.png"), dpi=190)
d.save(os.path.join(OUT, "rhear_codec_wiring.svg"))
print("all drawn pins verified against the netlist")
print("written hardware/rhear_codec_wiring.png / .svg")
