#!/usr/bin/env python3
"""RHEAR Phase 0 bench wiring, declared as a netlist and machine-checked.

Hand-drawn schematics hide mistakes. This declares every part and every net --
including the passives -- in SKiDL and runs its electrical rule check, so
unconnected pins, two outputs driving one net, and undriven power rails are
caught by a tool rather than by eye. The schematic in docs/ is generated from
this description, so the drawing cannot drift from the checked design.

Scope: the parts on Vellore Electronics bills 2917 / 2921 / 2935, the
MAX98357A, and the passives the design actually needs. The ADAU1772, its QFN40
adapter and the 12.000 MHz crystal ARE now here: the QFN40->DIP adapter is in
hand, so the codec is declared with its real ADAU1772BCPZ pinout and the eight
reserved GPIOs are wired instead of left open. The chip itself has not arrived;
soldering it to the adapter needs hot air because of the thermal pad (EP, pin
41), and until then the adapter simply sits empty in the socket.

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

# ------------------------------------------------- ADAU1772 on the QFN40 adapter
#
# Pinout is the ADAU1772BCPZ 40-LFCSP as printed on the datasheet symbol, NOT
# from memory. Pin 41 is the exposed thermal pad underneath the package: it is
# a GROUND connection and also the only path for heat out of the part, so it
# must be soldered, which is why this needs hot air rather than an iron.
adau = Part(name="ADAU1772BCPZ", tool=SKIDL, dest=TEMPLATE,
            description="ANC codec, 4 ADC / 2 DAC, 40-LFCSP on QFN40->DIP adapter",
            pins=[Pin(num="1",  name="SDA",      func=Pin.types.BIDIR),
                  Pin(num="2",  name="SCL",      func=Pin.types.INPUT),
                  Pin(num="3",  name="ADDR1",    func=Pin.types.INPUT),
                  Pin(num="4",  name="ADDR0",    func=Pin.types.INPUT),
                  Pin(num="5",  name="SELFBOOT", func=Pin.types.INPUT),
                  Pin(num="6",  name="MICBIAS0", func=Pin.types.PWROUT),
                  Pin(num="7",  name="MICBIAS1", func=Pin.types.PWROUT),
                  Pin(num="8",  name="AIN0REF",  func=Pin.types.INPUT),
                  Pin(num="9",  name="AIN0",     func=Pin.types.INPUT),
                  Pin(num="10", name="AVDD_1",   func=Pin.types.PWRIN),
                  Pin(num="11", name="AGND_1",   func=Pin.types.PWRIN),
                  Pin(num="12", name="CM",       func=Pin.types.PASSIVE),
                  Pin(num="13", name="AIN1REF",  func=Pin.types.INPUT),
                  Pin(num="14", name="AIN1",     func=Pin.types.INPUT),
                  Pin(num="15", name="AIN2REF",  func=Pin.types.INPUT),
                  Pin(num="16", name="AIN2",     func=Pin.types.INPUT),
                  Pin(num="17", name="AIN3REF",  func=Pin.types.INPUT),
                  Pin(num="18", name="AIN3",     func=Pin.types.INPUT),
                  Pin(num="19", name="AVDD_2",   func=Pin.types.PWRIN),
                  Pin(num="20", name="AGND_2",   func=Pin.types.PWRIN),
                  Pin(num="21", name="LOUTLN",   func=Pin.types.OUTPUT),
                  Pin(num="22", name="LOUTLP",   func=Pin.types.OUTPUT),
                  Pin(num="23", name="AGND_3",   func=Pin.types.PWRIN),
                  Pin(num="24", name="AVDD_3",   func=Pin.types.PWRIN),
                  Pin(num="25", name="LOUTRN",   func=Pin.types.OUTPUT),
                  Pin(num="26", name="LOUTRP",   func=Pin.types.OUTPUT),
                  Pin(num="27", name="PD",       func=Pin.types.INPUT),
                  Pin(num="28", name="REG_OUT",  func=Pin.types.PWROUT),
                  Pin(num="29", name="DVDD",     func=Pin.types.PWRIN),
                  Pin(num="30", name="DGND",     func=Pin.types.PWRIN),
                  Pin(num="31", name="LRCLK",    func=Pin.types.BIDIR),
                  Pin(num="32", name="BCLK",     func=Pin.types.BIDIR),
                  Pin(num="33", name="DAC_SDATA", func=Pin.types.INPUT),
                  Pin(num="34", name="ADC_SDATA0", func=Pin.types.OUTPUT),
                  Pin(num="35", name="ADC_SDATA1", func=Pin.types.OUTPUT),
                  Pin(num="36", name="DMIC2_3",  func=Pin.types.BIDIR),
                  Pin(num="37", name="DMIC0_1",  func=Pin.types.BIDIR),
                  Pin(num="38", name="XTALO",    func=Pin.types.OUTPUT),
                  Pin(num="39", name="XTALI",    func=Pin.types.INPUT),
                  Pin(num="40", name="IOVDD",    func=Pin.types.PWRIN),
                  Pin(num="41", name="EP",       func=Pin.types.PWRIN)])

xtal = Part(name="XTAL_12M", tool=SKIDL, dest=TEMPLATE,
            description="12.000 MHz crystal (KNS), ADAU1772 XTALI/XTALO",
            pins=[Pin(num="1", name="A", func=Pin.types.PASSIVE),
                  Pin(num="2", name="B", func=Pin.types.PASSIVE)])

U4 = adau(ref="U4")

# --- supplies. AVDD/IOVDD accept 1.8-3.3 V; DVDD is fed from the part's own
#     on-board regulator via REG_OUT, NOT from the 3V3 rail.
for pin in ("AVDD_1", "AVDD_2", "AVDD_3", "IOVDD"):
    U4[pin] += V33
for pin in ("AGND_1", "AGND_2", "AGND_3", "DGND", "EP"):
    U4[pin] += GND
U4["DVDD"] += U4["REG_OUT"]          # internal regulator supplies the digital core

# --- 12.000 MHz crystal across XTALI/XTALO with its two load capacitors.
#     12.000 MHz is inside the part's 8-27 MHz crystal-amplifier range; the PLL
#     multiplies it to the 24.576 MHz the audio clocks need.
Y1 = xtal(ref="Y1", value="12.000 MHz")
U4["XTALI"] += Y1[1]
U4["XTALO"] += Y1[2]
for ref, pin in (("C12", Y1[1]), ("C13", Y1[2])):
    c = cap(ref=ref, value="22 pF")  # load caps, one per crystal leg
    c[1] += pin
    c[2] += GND

# --- I2C to the ESP32, with the pull-ups the datasheet specifies (2.0 kohm,
#     not the 4.7 kohm rule of thumb).
U4["SDA"] += U1["GPIO9"]
U4["SCL"] += U1["GPIO10"]
for ref, net in (("R5", U4["SDA"]), ("R6", U4["SCL"])):
    r = res(ref=ref, value="2.0 kohm")
    r[1] += V33
    r[2] += net

# --- I2C address = base: both select pins low. SELFBOOT low so the ESP32
#     configures the part over I2C instead of it booting from an EEPROM.
U4["ADDR0"] += GND
U4["ADDR1"] += GND
U4["SELFBOOT"] += GND
U4["PD"] += U1["GPIO11"]             # power-down / reset under host control

# --- I2S on the ESP32's peripheral 0. The INMP441 keeps peripheral 1.
U4["BCLK"]       += U1["GPIO12"]
U4["LRCLK"]      += U1["GPIO13"]
U4["DAC_SDATA"]  += U1["GPIO14"]     # ESP32 -> codec (anti-noise out)
U4["ADC_SDATA0"] += U1["GPIO21"]     # codec -> ESP32 (four mic channels)

# --- decoupling: one 100 nF per supply pin, plus REG_OUT and CM.
for ref, pin in (("C14", U4["AVDD_1"]), ("C15", U4["AVDD_2"]),
                 ("C16", U4["AVDD_3"]), ("C17", U4["IOVDD"]),
                 ("C18", U4["REG_OUT"]), ("C19", U4["CM"])):
    c = cap(ref=ref, value="100 nF")
    c[1] += pin
    c[2] += GND

# --- the four analog mics move onto the codec's differential inputs. Each
#     MAX4466 is single-ended, so its REF pin sits at the codec's own common
#     mode. In phase 0 they stay on the ESP32 ADC; these are the phase-1 seats.
no_connect(U4["AIN0"], U4["AIN0REF"], U4["AIN1"], U4["AIN1REF"],
           U4["AIN2"], U4["AIN2REF"], U4["AIN3"], U4["AIN3REF"])
no_connect(U4["LOUTLN"], U4["LOUTLP"], U4["LOUTRN"], U4["LOUTRP"])
no_connect(U4["MICBIAS0"], U4["MICBIAS1"])
no_connect(U4["ADC_SDATA1"], U4["DMIC0_1"], U4["DMIC2_3"])
no_connect(U1["GPIO8"])              # codec MCLK not needed: the crystal drives it

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
