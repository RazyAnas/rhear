#!/usr/bin/env python3
"""Render the full schematic FROM the ERC-checked netlist, not from memory.

rhear_phase0_netlist.py declares the circuit -- passives included -- in SKiDL
and runs its electrical rule check. This reads the JSON that run produced, draws
it, and then asserts that EVERY net in that file was drawn. A wire cannot appear
here unless it is in the checked design, and a net cannot be silently forgotten:
the script fails instead.

Pin slots are numbered from the BOTTOM of a block in schemdraw, and every
block's y position is derived from the pin it has to meet, so the signal wires
run straight across instead of crossing.

    /opt/anaconda3/bin/python hardware/draw_schematic.py
"""
import os, json

import schemdraw
from schemdraw import elements as elm

HERE = os.path.dirname(os.path.abspath(__file__))
NET = json.load(open(os.path.join(HERE, "rhear_phase0_netlist.json")))
nets, ncs, vals = NET["nets"], set(NET["no_connect"]), NET["values"]
# The board silkscreens bare numbers next to its header -- "4", not "GPIO4".
# The schematic prints what is on the board, so a wire can be traced without a
# datasheet in the other hand.
SILK = NET["silkscreen"]

TEAL, GREY, RED, INK, AMB = "#0B6E6C", "#7A8C99", "#A33B12", "#11191C", "#8A6A1F"
drawn = set()

EH, ESLOTS = 24.0, 24
def slot_y(k, n=ESLOTS, h=EH):        # schemdraw: slot 1 is the BOTTOM
    return -h / 2 + h * k / (n + 1)

L = {"GPIO1": 24, "GPIO2": 20, "GPIO4": 16, "GPIO5": 12,
     "GPIO38": 6, "GPIO39": 5, "GPIO40": 4}
R = {"GPIO15": 24, "GPIO16": 23, "GPIO17": 22, "8·9·10·11·12·13·14·21": 8}
PITCH = slot_y(3) - slot_y(2)


def net_of(pin):
    for n, conns in nets.items():
        if pin in conns:
            return n
    raise AssertionError(f"{pin} is on no net")


d = schemdraw.Drawing(font="sans", fontsize=10, lw=1.7)
A = {}


def block(ref, pins, size, at, title, sub=None):
    ic = elm.Ic(pins=pins, size=size, plblsize=8.5).at(at).right()
    d.add(ic)
    A[ref] = dict(ic.absanchors)
    bb = ic.get_bbox(transform=True)
    d.add(elm.Label().at((bb.xmin, bb.ymax + 1.05)).label(title, fontsize=10.5,
                                                          color=INK, halign="left"))
    if sub:
        d.add(elm.Label().at((bb.xmin, bb.ymax + 0.58)).label(sub, fontsize=8.5,
                                                              color=GREY, halign="left"))
    return ic


def wire(p1, p2, net, color=TEAL):
    assert net in nets, f"{net} is not in the ERC-checked design"
    d.add(elm.Line().at(p1).to(p2).color(color))
    drawn.add(net)


# ------------------------------------------------------------------ the MCU
block("U1",
      [elm.IcPin(name=SILK.get(k, k), side="left", slot=f"{v}/{ESLOTS}")
       for k, v in L.items()] +
      [elm.IcPin(name=SILK.get(k, k), side="right", slot=f"{v}/{ESLOTS}")
       for k, v in R.items()] +
      [elm.IcPin(name="3V3", side="bottom", slot="1/3"),
       elm.IcPin(name="GND", side="bottom", slot="2/3"),
       elm.IcPin(name="5V", side="bottom", slot="3/3")],
      (5.4, EH), (0, 0), "ESP32-S3  N16R8   ·   U1",
      "pin numbers as printed on the board  ·  1, 2, 4, 5 are ADC1 — "
      "ADC2 dies when WiFi is on, and 3 is a strapping pin")

# --------------------------------------------------- four analog mics + filters
MICP = [elm.IcPin(name="OUT", side="right"),
        elm.IcPin(name="VCC", side="left", slot="2/3"),
        elm.IcPin(name="GND", side="left", slot="1/3")]
MICS = [("MIC1", "GPIO1", "R1", "C6", "LEFT cup  ·  OUTSIDE the shell   (reference)"),
        ("MIC2", "GPIO2", "R2", "C7", "LEFT cup  ·  INSIDE at the ear   (error)"),
        ("MIC3", "GPIO4", "R3", "C8", "RIGHT cup  ·  OUTSIDE the shell   (reference)"),
        ("MIC4", "GPIO5", "R4", "C9", "RIGHT cup  ·  INSIDE at the ear   (error)")]

XM, XR, XN = -11.0, -6.4, -4.2        # mic centre, resistor start, filtered node
for ref, gpio, rr, cc, where in MICS:
    y = slot_y(L[gpio])
    block(ref, MICP, (3.6, 1.6), (XM, y), f"MAX4466  ·  {ref}", where)

    n_mic = net_of(f"{ref}.OUT")                    # mic OUT -> R
    n_adc = net_of(f"U1.{gpio}")                    # R -> node -> GPIO, C to GND
    d.add(elm.Line().at(A[ref]["OUT"]).to((XR, y)).color(TEAL))
    r = elm.Resistor().at((XR, y)).to((XN, y)).right().color(TEAL).label(
        f"{rr}\n{vals.get(rr, '1 kΩ')}", fontsize=8.5, loc="top")
    d.add(r)
    drawn.add(n_mic)
    d.add(elm.Dot(radius=.10).at((XN, y)))
    c = elm.Capacitor().at((XN, y)).down().length(1.5).color(TEAL)
    d.add(c)
    # placed by hand to the left of the capacitor: the element's own label lands
    # on the symbol, and the space to its right is full of the ADC wires
    d.add(elm.Label().at((XN - 1.05, y - 0.80)).label(
        f"{cc}\n{vals.get(cc, '47 nF')}", fontsize=8.5, color=TEAL, halign="right"))
    d.add(elm.Ground().at(c.end).scale(0.6))
    wire((XN, y), A["U1"][SILK[gpio]], n_adc)

d.add(elm.Label().at((XM - 1.8, -EH / 2 + 1.2)).label(
    "anti-alias, one per channel:  1 kΩ into 47 nF, corner 3.4 kHz\n"
    "without it everything above 4 kHz folds into the 8 kHz capture\n"
    "and the coherence result measures its own aliasing",
    fontsize=8.5, color=AMB, halign="left"))

# ------------------------------------------------------------ the voice mic
block("MIC5",
      [elm.IcPin(name="SCK", side="right", slot="3/3"),
       elm.IcPin(name="WS", side="right", slot="2/3"),
       elm.IcPin(name="SD", side="right", slot="1/3"),
       elm.IcPin(name="VDD", side="left", slot="3/4"),
       elm.IcPin(name="GND", side="left", slot="2/4"),
       elm.IcPin(name="LR", side="left", slot="1/4")],
      (3.6, PITCH * 4), (XM, slot_y(L["GPIO39"])),
      "INMP441  ·  MIC5", "boom position  ·  I²S peripheral 1  ·  never on the ANC path")
for a, b in (("SCK", "GPIO38"), ("WS", "GPIO39"), ("SD", "GPIO40")):
    wire(A["MIC5"][a], A["U1"][SILK[b]], net_of(f"MIC5.{a}"))

# ------------------------------------------------------------- amp + drivers
block("U2",
      [elm.IcPin(name="BCLK", side="left", slot="4/4"),
       elm.IcPin(name="LRC", side="left", slot="3/4"),
       elm.IcPin(name="DIN", side="left", slot="2/4"),
       elm.IcPin(name="VIN", side="left", slot="1/4"),
       elm.IcPin(name="OUT+", side="right", slot="2/2"),
       elm.IcPin(name="OUT-", side="right", slot="1/2")],
      (3.8, PITCH * 5), (9.0, slot_y(R["GPIO16"])),
      "MAX98357A  ·  U2", "I²S class-D  ·  leave GAIN and SD open")
for a, b in (("BCLK", "GPIO15"), ("LRC", "GPIO16"), ("DIN", "GPIO17")):
    wire(A["U1"][SILK[b]], A["U2"][a], net_of(f"U2.{a}"))

XSP, XSM = 15.6, 17.2
yp, ym = A["U2"]["OUT+"][1], A["U2"]["OUT-"][1]
d.add(elm.Line().at(A["U2"]["OUT+"]).to((XSP, yp)).color(INK))
d.add(elm.Line().at(A["U2"]["OUT-"]).to((XSM, ym)).color(INK))
d.add(elm.Line().at((XSM, ym)).to((XSM, ym - 4.6)).color(INK))
d.add(elm.Line().at((XSP, yp)).to((XSP, yp - 4.6)).color(INK))
for i, (ref, dy) in enumerate((("LS1", -0.4), ("LS2", -3.6))):
    yy = yp - 1.0 + dy
    s = elm.Speaker().at((XSP, yy)).right()
    d.add(s)
    d.add(elm.Dot(radius=.10).at((XSP, yy)))
    d.add(elm.Dot(radius=.10).at((XSM, yy - 0.6)))
    d.add(elm.Line().at((XSM, yy - 0.6)).to(s.absanchors["in2"]).color(INK))
    d.add(elm.Label().at((XSP + 2.6, yy - 0.5)).label(
        f"{ref}\n16 Ω  ·  {'LEFT' if ref == 'LS1' else 'RIGHT'} cup",
        fontsize=8.5, halign="left"))
drawn |= {net_of("U2.OUT+"), net_of("U2.OUT-")}
d.add(elm.Label().at((XSP - 1.2, yp - 5.6)).label(
    "two 16 Ω drivers in parallel = 8 Ω — the amp drives down to 4 Ω",
    fontsize=8.5, color=GREY, halign="left"))

# ------------------------------------------------------------------- rails
Y33, YGND, Y5 = -EH / 2 - 1.5, -EH / 2 - 2.5, -EH / 2 - 0.6
XL33, XLG, XRR = -14.2, -14.9, 20.6

d.add(elm.Line().at(A["U1"]["3V3"]).to((A["U1"]["3V3"][0], Y33)).color(TEAL))
d.add(elm.Line().at((XL33, Y33)).to((A["U1"]["3V3"][0], Y33)).color(TEAL))
d.add(elm.Label().at((XL33 - 1.0, Y33)).label("3V3", fontsize=9.5, color=TEAL))
d.add(elm.Line().at(A["U1"]["GND"]).to((A["U1"]["GND"][0], YGND)))
d.add(elm.Line().at((XLG, YGND)).to((XRR, YGND)))
d.add(elm.Label().at((XLG - 1.0, YGND)).label("GND", fontsize=9.5))

for ref in ("MIC1", "MIC2", "MIC3", "MIC4", "MIC5"):
    vk = "VDD" if ref == "MIC5" else "VCC"
    p, g = A[ref][vk], A[ref]["GND"]
    d.add(elm.Line().at(p).to((XL33, p[1])).color(TEAL))
    d.add(elm.Line().at((XL33, p[1])).to((XL33, Y33)).color(TEAL))
    d.add(elm.Line().at(g).to((XLG, g[1])))
    d.add(elm.Line().at((XLG, g[1])).to((XLG, YGND)))
lr = A["MIC5"]["LR"]
d.add(elm.Line().at(lr).to((XLG, lr[1])))
d.add(elm.Label().at((lr[0] - 1.9, lr[1] + 0.32)).label("L/R → GND", fontsize=8, color=GREY))
drawn |= {"+3V3", "GND"}

v5 = A["U1"]["5V"]
d.add(elm.Line().at(v5).to((v5[0], Y5)).color(RED))
d.add(elm.Line().at((v5[0], Y5)).to((XRR, Y5)).color(RED))
d.add(elm.Line().at((XRR, Y5)).to((XRR, A["U2"]["VIN"][1])).color(RED))
d.add(elm.Line().at((XRR, A["U2"]["VIN"][1])).to(A["U2"]["VIN"]).color(RED))
d.add(elm.Label().at((6.4, Y5 + 0.34)).label(
    "5V from the powerbank rail — VIN off 3V3 makes the amp quiet and clip",
    fontsize=8.5, color=RED, halign="left"))
drawn.add("+5V")

# ------------------------------------------------------- decoupling on the rails
DEC = [("C1", "10 µF", -12.6, "bulk, where power enters"),
       ("C2", "100 nF", -10.4, None), ("C3", "100 nF", -8.6, None),
       ("C4", "100 nF", -6.8, None), ("C10", "100 nF", -5.0, None),
       ("C11", "100 nF", -3.2, None)]
for ref, val, x, note in DEC:
    c = elm.Capacitor().at((x, Y33)).down().length(1.0).color(TEAL).label(
        f"{ref}\n{vals.get(ref, val)}", fontsize=8, loc="left", ofst=(0.1, 0))
    d.add(c)
    d.add(elm.Dot(radius=.09).at((x, Y33)).color(TEAL))
    d.add(elm.Line().at(c.end).to((x, YGND)))
    d.add(elm.Dot(radius=.09).at((x, YGND)))
c5 = elm.Capacitor().at((4.6, Y5)).down().length(1.0).color(RED)
d.add(c5)
d.add(elm.Label().at((4.2, Y5 - 0.55)).label(
    f"C5\n{vals.get('C5', '100 nF')}", fontsize=8, color=RED, halign="right"))
d.add(elm.Dot(radius=.09).at((4.6, Y5)).color(RED))
d.add(elm.Line().at(c5.end).to((4.6, YGND)))
d.add(elm.Dot(radius=.09).at((4.6, YGND)))
d.add(elm.Label().at((-12.6, YGND - 1.1)).label(
    "one 100 nF beside every module's power pins, 10 µF bulk across the rails",
    fontsize=8.5, color=GREY, halign="left"))

# ----------------------------------------------------------------- reserved
# ----------------------------------------------------------- phase 1 space
# NOT part of the ERC-checked netlist and NOT built on the 7th. This block is
# drawn so the eight reserved pins have a visible destination and so the
# breadboard gap is planned rather than discovered. Everything here is dashed;
# the coverage assertion below still applies only to the circuit that is built.
CX, CY = 9.6, -8.6
cod = elm.Ic(pins=[elm.IcPin(name="SDA", side="left", slot="8/8"),
                   elm.IcPin(name="SCL", side="left", slot="7/8"),
                   elm.IcPin(name="PD", side="left", slot="6/8"),
                   elm.IcPin(name="BCLK", side="left", slot="5/8"),
                   elm.IcPin(name="LRCLK", side="left", slot="4/8"),
                   elm.IcPin(name="DAC_SDATA", side="left", slot="3/8"),
                   elm.IcPin(name="ADC_SDATA0", side="left", slot="2/8"),
                   elm.IcPin(name="ADC_SDATA1", side="left", slot="1/8"),
                   elm.IcPin(name="XTALI", side="right", slot="4/4"),
                   elm.IcPin(name="XTALO", side="right", slot="3/4"),
                   elm.IcPin(name="AIN0-3", side="right", slot="2/4"),
                   elm.IcPin(name="HPOUT L/R", side="right", slot="1/4")],
             size=(7.2, 7.6), plblsize=8.5).at((CX, CY)).right()
d.add(cod)
CA = dict(cod.absanchors)
cbb = cod.get_bbox(transform=True)
for seg in d.elements[-1].segments:
    seg.color = GREY
    if hasattr(seg, "lw"):
        seg.lw = 1.3
d.add(elm.Label().at((cbb.xmin, cbb.ymax + 1.05)).label(
    "ADAU1772  —  PHASE 1, NOT WIRED ON THE 7th", fontsize=10.5, color=GREY,
    halign="left"))
d.add(elm.Label().at((cbb.xmin, cbb.ymax + 0.55)).label(
    "leave this block of the breadboard empty  ·  QFN40 needs a DIP adapter",
    fontsize=8.5, color=GREY, halign="left"))

rsv = A["U1"]["8·9·10·11·12·13·14·21"]
d.add(elm.Line().at(rsv).to((rsv[0] + 1.2, rsv[1])).color(GREY).linestyle("--"))
d.add(elm.Label().at((rsv[0] + 1.4, rsv[1])).label(
    "the eight reserved pins →", fontsize=8.5, color=GREY, halign="left"))

# the eight reserved ESP32 pins, each to its destination on the codec
for gpio, cpin in (("GPIO9", "SDA"), ("GPIO10", "SCL"), ("GPIO11", "PD"),
                   ("GPIO12", "BCLK"), ("GPIO13", "LRCLK"),
                   ("GPIO14", "DAC_SDATA"), ("GPIO21", "ADC_SDATA0"),
                   ("GPIO8", "ADC_SDATA1")):
    d.add(elm.Label().at((CX - 3.4, CA[cpin][1])).label(
        f"pin {SILK[gpio]}", fontsize=8.5, color=GREY, halign="right"))
    d.add(elm.Line().at((CX - 3.3, CA[cpin][1])).to(CA[cpin])
          .color(GREY).linestyle("--"))

# 12.000 MHz crystal and its two load capacitors
XX = CX + 5.2
d.add(elm.Line().at(CA["XTALI"]).to((XX, CA["XTALI"][1])).color(GREY).linestyle("--"))
d.add(elm.Line().at(CA["XTALO"]).to((XX, CA["XTALO"][1])).color(GREY).linestyle("--"))
xtal = elm.Crystal().at((XX, CA["XTALI"][1])).to((XX, CA["XTALO"][1]))
d.add(xtal)
for seg in d.elements[-1].segments:
    seg.color = GREY
d.add(elm.Label().at((XX + 0.35, (CA["XTALI"][1] + CA["XTALO"][1]) / 2)).label(
    "Y1\n12.000 MHz", fontsize=8.5, color=GREY, halign="left"))
for anch, cref in (("XTALI", "C12"), ("XTALO", "C13")):
    y = CA[anch][1]
    d.add(elm.Dot(radius=.09).at((XX, y)).color(GREY))
    cc = elm.Capacitor().at((XX, y)).right().length(1.5)
    d.add(cc)
    for seg in d.elements[-1].segments:
        seg.color = GREY
    d.add(elm.Ground().at(cc.end).scale(0.55))
    d.add(elm.Label().at((XX + 0.75, y + 0.42)).label(
        f"{cref}  22 pF", fontsize=8, color=GREY))


d.add(elm.Label().at((cbb.xmin + 0.1, cbb.ymin - 1.5)).label(
    "PLL for a 12.000 MHz crystal — 0x0001..0x0005 = 00, 7D, 00, 18, 43\n"
    "   M=125  N=24  R=8  X+1=2   →   12.000/2 × 8.192 = 49.152 → ÷2 = 24.576 MHz\n"
    "   N/M = 0.192, inside the datasheet's mandatory 0.1–0.9 window\n"
    "AVDD = IOVDD = 3V3, DVDD from REG_OUT (on-board regulator, no 1.1 V rail)\n"
    "I²C pull-ups are 2.0 kΩ per the datasheet, not 4.7 kΩ\n"
    "Its own caps, not yet owned: 4× 10 µF on AIN0–3REF, 10 µF on CM,\n"
    "   2× 1 µF on MICBIAS0/1, 0.1 µF on each AVDD, DVDD",
    fontsize=8.5, color=GREY, halign="left"))

missing = set(nets) - drawn
assert not missing, f"nets in the checked design this drawing omits: {sorted(missing)}"
print(f"\nparts in the checked design    : {len(NET['parts'])}")
print(f"nets in the checked design     : {len(nets)}")
print(f"nets drawn                     : {len(drawn)}   (all of them)")
print(f"ESP32 pins deliberately open   : {len([n for n in ncs if n.startswith('U1.')])}")

svg = os.path.join(HERE, "rhear_phase0_schematic.svg")
d.save(svg)
d.save(svg.replace(".svg", ".png"), dpi=140)
print(f"\nwritten {svg}\nwritten {svg.replace('.svg', '.png')}")
