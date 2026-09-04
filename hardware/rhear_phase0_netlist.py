#!/usr/bin/env python3
"""RHEAR Phase 0 bench wiring, declared as a netlist and machine-checked.

Hand-drawn schematics hide mistakes. This declares every part and every net --
including the passives -- in SKiDL and runs its electrical rule check, so
unconnected pins, two outputs driving one net, and undriven power rails are
caught by a tool rather than by eye. The schematic in docs/ is generated from
this description, so the drawing cannot drift from the checked design.

Scope: the parts on Vellore Electronics bills 2917 / 2921 / 2935, the
MAX98357A, and the passives the design actually needs. The ADAU1772, its QFN40
adapter and the 12.288 MHz crystal are NOT here -- their eight GPIOs are
deliberately left open, declared as intentional no-connects.

WHY THE RC FILTERS EXIST. The two ANC microphones feed the ESP32's own ADC,
sampled at 8 kHz, so anything above 4 kHz folds back into the measurement band.
The coherence result -- the whole point of the analog front end -- would be
corrupted by its own aliasing. Each channel therefore gets a single-pole
anti-alias filter, 1 kohm into 47 nF, corner 3.4 kHz. One pole gives about 7 dB
of rejection at 8 kHz, which is enough to trust the 100 Hz - 2 kHz bands where
the cancellation bound actually matters, and not enough to treat 2 - 4 kHz as
more than indicative. The series resistor also limits current into the ADC's
sample-and-hold, which is why it sits on the microphone side.

    /opt/anaconda3/bin/python hardware/rhear_phase0_netlist.py
"""
import os, json

from skidl import (Part, Net, Pin, SKIDL, TEMPLATE, ERC, set_default_tool)

set_default_tool(SKIDL)
OUT = os.path.dirname(os.path.abspath(__file__))


def no_connect(*pins):
    """Mark a pin as an INTENTIONAL no-connect.

    SKiDL 2.3 dropped the importable NC net; silencing ERC per pin is the
    portable equivalent and records the intent in the source rather than hiding
    a real omission. Everything marked here is either a codec pin that stays
    empty until the ADAU1772 arrives, or a breakout pin the board's own internal
    divider already sets."""
    for pin in pins:
        pin.do_erc = False


# ---------------------------------------------------------------- parts
esp = Part(name="ESP32-S3-DevKitC-1", tool=SKIDL, dest=TEMPLATE,
           description="ESP32-S3 N16R8 dev board (bill 2917 item 1)",
           pins=[Pin(num="1", name="3V3", func=Pin.types.PWROUT),
                 Pin(num="2", name="5V", func=Pin.types.PWROUT),
                 Pin(num="3", name="GND", func=Pin.types.PWRIN),
                 Pin(num="4", name="GPIO1", func=Pin.types.INPUT),
                 Pin(num="5", name="GPIO2", func=Pin.types.INPUT),
                 Pin(num="6", name="GPIO4", func=Pin.types.INPUT),
                 Pin(num="7", name="GPIO5", func=Pin.types.INPUT),
                 Pin(num="8", name="GPIO38", func=Pin.types.OUTPUT),
                 Pin(num="9", name="GPIO39", func=Pin.types.OUTPUT),
                 Pin(num="21", name="GPIO40", func=Pin.types.INPUT),
                 Pin(num="10", name="GPIO15", func=Pin.types.OUTPUT),
                 Pin(num="11", name="GPIO16", func=Pin.types.OUTPUT),
                 Pin(num="12", name="GPIO17", func=Pin.types.OUTPUT),
                 Pin(num="13", name="GPIO8", func=Pin.types.BIDIR),
                 Pin(num="14", name="GPIO9", func=Pin.types.BIDIR),
                 Pin(num="15", name="GPIO10", func=Pin.types.BIDIR),
                 Pin(num="16", name="GPIO11", func=Pin.types.BIDIR),
                 Pin(num="17", name="GPIO12", func=Pin.types.BIDIR),
                 Pin(num="18", name="GPIO13", func=Pin.types.BIDIR),
                 Pin(num="19", name="GPIO14", func=Pin.types.BIDIR),
                 Pin(num="20", name="GPIO21", func=Pin.types.BIDIR)])

max4466 = Part(name="MAX4466", tool=SKIDL, dest=TEMPLATE,
               description="electret mic + adjustable amp (bills 2917/2935, 8 off)",
               pins=[Pin(num="1", name="VCC", func=Pin.types.PWRIN),
                     Pin(num="2", name="GND", func=Pin.types.PWRIN),
                     Pin(num="3", name="OUT", func=Pin.types.OUTPUT)])

inmp441 = Part(name="INMP441", tool=SKIDL, dest=TEMPLATE,
               description="I2S MEMS microphone (bill 2917 item 2)",
               pins=[Pin(num="1", name="VDD", func=Pin.types.PWRIN),
                     Pin(num="2", name="GND", func=Pin.types.PWRIN),
                     Pin(num="3", name="SCK", func=Pin.types.INPUT),
                     Pin(num="4", name="WS", func=Pin.types.INPUT),
                     Pin(num="5", name="SD", func=Pin.types.OUTPUT),
                     Pin(num="6", name="LR", func=Pin.types.INPUT)])

max98357 = Part(name="MAX98357A", tool=SKIDL, dest=TEMPLATE,
                description="I2S class-D amplifier breakout",
                pins=[Pin(num="1", name="VIN", func=Pin.types.PWRIN),
                      Pin(num="2", name="GND", func=Pin.types.PWRIN),
                      Pin(num="3", name="BCLK", func=Pin.types.INPUT),
                      Pin(num="4", name="LRC", func=Pin.types.INPUT),
                      Pin(num="5", name="DIN", func=Pin.types.INPUT),
                      Pin(num="6", name="OUT+", func=Pin.types.OUTPUT),
                      Pin(num="7", name="OUT-", func=Pin.types.OUTPUT),
                      Pin(num="8", name="GAIN", func=Pin.types.INPUT),
                      Pin(num="9", name="SD", func=Pin.types.INPUT)])

spk = Part(name="SPEAKER_16R", tool=SKIDL, dest=TEMPLATE,
           description="16 ohm 0.25 W 35 mm driver (bill 2917 item 4, 2 off)",
           pins=[Pin(num="1", name="+", func=Pin.types.PASSIVE),
                 Pin(num="2", name="-", func=Pin.types.PASSIVE)])

res = Part(name="R", tool=SKIDL, dest=TEMPLATE, description="resistor",
           pins=[Pin(num="1", name="1", func=Pin.types.PASSIVE),
                 Pin(num="2", name="2", func=Pin.types.PASSIVE)])

cap = Part(name="C", tool=SKIDL, dest=TEMPLATE, description="capacitor",
           pins=[Pin(num="1", name="1", func=Pin.types.PASSIVE),
                 Pin(num="2", name="2", func=Pin.types.PASSIVE)])

eeprom = Part(name="AT24C256", tool=SKIDL, dest=TEMPLATE,
              description="I2C EEPROM module (bill 2917 item 9) -- PHASE 1",
              pins=[Pin(num="1", name="VCC", func=Pin.types.PWRIN),
                    Pin(num="2", name="GND", func=Pin.types.PWRIN),
                    Pin(num="3", name="SDA", func=Pin.types.BIDIR),
                    Pin(num="4", name="SCL", func=Pin.types.INPUT)])

# ---------------------------------------------------------------- rails
V33 = Net("+3V3"); V33.drive = Pin.drives.POWER
V5  = Net("+5V");  V5.drive  = Pin.drives.POWER
GND = Net("GND");  GND.drive = Pin.drives.POWER

U1 = esp(ref="U1")
U1["3V3"] += V33
U1["5V"]  += V5
U1["GND"] += GND

# ------------------------------------------------- analog mics + anti-alias
# Four analog microphones: a reference OUTSIDE and an error INSIDE each cup.
# That is the real feed-forward ANC topology, one pair per ear, and it is what
# makes the coherence measurement a per-cup result rather than a single number.
#
# All four sit on ADC1. ADC2 stops working the moment WiFi comes up, and GPIO3
# is a strapping pin, so the four usable ADC1 channels here are 1, 2, 4 and 5 --
# which is why the INMP441 moved off 4/5/6 to 38/39/40 (also the JTAG pins, free
# because the board is flashed over USB).
ANALOG = [("MIC1", "GPIO1", "R1", "C6", "LEFT cup, OUTSIDE the shell"),
          ("MIC2", "GPIO2", "R2", "C7", "LEFT cup, INSIDE at the ear"),
          ("MIC3", "GPIO4", "R3", "C8", "RIGHT cup, OUTSIDE the shell"),
          ("MIC4", "GPIO5", "R4", "C9", "RIGHT cup, INSIDE at the ear")]
for ref, gpio, rref, cref, where in ANALOG:
    m = max4466(ref=ref)
    m["VCC"] += V33
    m["GND"] += GND
    r = res(ref=rref, value="1 kΩ")
    c = cap(ref=cref, value="47 nF")
    m["OUT"] += r[1]                 # series into the ADC
    r[2] += U1[gpio]                 # filtered node IS the ADC pin
    c[1] += U1[gpio]
    c[2] += GND                      # shunt to ground: fc = 3.4 kHz

# ------------------------------------------------- voice mic, I2S peripheral 1
VOX = inmp441(ref="MIC5")
VOX["VDD"] += V33
VOX["GND"] += GND
VOX["LR"]  += GND                    # left channel
VOX["SCK"] += U1["GPIO38"]
VOX["WS"]  += U1["GPIO39"]
VOX["SD"]  += U1["GPIO40"]

# ------------------------------------------------- audio out, I2S peripheral 0
U2 = max98357(ref="U2")
U2["VIN"] += V5                      # 5 V, not 3V3 -- quiet and clipping otherwise
U2["GND"] += GND
U2["BCLK"] += U1["GPIO15"]
U2["LRC"]  += U1["GPIO16"]
U2["DIN"]  += U1["GPIO17"]
no_connect(U2["GAIN"], U2["SD"])     # breakout's own dividers: 9 dB, enabled, (L+R)/2

# Two 16 ohm drivers in parallel present 8 ohm; the amp drives down to 4 ohm.
LS1 = spk(ref="LS1"); LS2 = spk(ref="LS2")
LS1["+"] += U2["OUT+"]; LS1["-"] += U2["OUT-"]
LS2["+"] += U2["OUT+"]; LS2["-"] += U2["OUT-"]

# ------------------------------------------------- decoupling
C1 = cap(ref="C1", value="10 µF")      # bulk, where power enters the rails
C1[1] += V33; C1[2] += GND
for ref, rail in (("C2", V33), ("C3", V33), ("C4", V33),
                  ("C10", V33), ("C11", V33), ("C5", V5)):
    c = cap(ref=ref, value="100 nF")   # one beside each module's power pins
    c[1] += rail; c[2] += GND

# ------------------------------------------------- phase 1, deliberately open
RESERVED = ["GPIO8", "GPIO9", "GPIO10", "GPIO11",
            "GPIO12", "GPIO13", "GPIO14", "GPIO21"]
for p in RESERVED:
    no_connect(U1[p])

U3 = eeprom(ref="U3")                # shares the codec's I2C bus, so it waits
no_connect(U3["VCC"], U3["GND"], U3["SDA"], U3["SCL"])

# ---------------------------------------------------------------- check
print("\n" + "=" * 66)
print("ELECTRICAL RULE CHECK")
print("=" * 66)
ERC()

ckt = U1.circuit
netlist = {}
for n in ckt.nets:
    conns = sorted(f"{pin.part.ref}.{pin.name}" for pin in n.pins)
    if conns:
        netlist[n.name] = conns
nc = sorted(f"{pin.part.ref}.{pin.name}"
            for part in ckt.parts for pin in part.pins if not pin.do_erc)

print("\n" + "=" * 66)
print("NETLIST")
print("=" * 66)
for name in sorted(netlist):
    print(f"  {name:12s}  {', '.join(netlist[name])}")
print(f"\nINTENTIONAL NO-CONNECTS ({len(nc)}):")
for x in nc:
    print(f"  {x}")

vals = {p.ref: getattr(p, "value", "") for p in ckt.parts}

# What is silkscreened on the ESP32-S3-DevKitC-1 header next to each pin. The
# board prints bare numbers, not "GPIO4" -- so that is what the schematic shows.
silk = {"GPIO1": "1", "GPIO2": "2", "GPIO4": "4", "GPIO5": "5",
        "GPIO38": "38", "GPIO39": "39", "GPIO40": "40",
        "GPIO15": "15", "GPIO16": "16", "GPIO17": "17",
        "GPIO8": "8", "GPIO9": "9", "GPIO10": "10", "GPIO11": "11",
        "GPIO12": "12", "GPIO13": "13", "GPIO14": "14", "GPIO21": "21",
        "3V3": "3V3", "5V": "5V", "GND": "GND"}
out = os.path.join(OUT, "rhear_phase0_netlist.json")
with open(out, "w") as f:
    json.dump({"nets": netlist, "no_connect": nc, "values": vals,
               "silkscreen": silk,
               "parts": {p.ref: p.description for p in ckt.parts}}, f, indent=1)
print(f"\nparts {len(ckt.parts)} | nets {len(netlist)}")
print(f"written {out}")
