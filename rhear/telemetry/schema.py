"""RHEAR telemetry contract.

THE POINT OF THIS FILE: the dashboard renders whatever a producer publishes,
generically. A new experiment, a laptop rig, a dev board or the final headset all
emit the same TelemetryFrame, so the visualisation is identical across the whole
programme and no frontend change is needed to surface a new quantity.

Rules for producers:
  * Every number must come from the running pipeline. Never synthesise a value to
    make a panel look populated. If a quantity is not measured, omit it -- the UI
    renders "--" and marks the panel stale, which is the honest display.
  * Publish at whatever rate you like; the bus decimates for the browser.
  * Add new quantities by adding entries to `scalars` / `series` / `vectors`.
    Do not add new top-level fields without bumping SCHEMA_VERSION.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal
import time

SCHEMA_VERSION = "1.0.0"

SourceKind = Literal["sim", "laptop", "devboard", "headset"]
SystemMode = Literal["NORMAL", "ENGINE", "ROTOR", "WIND", "IMPULSE",
                     "PROTECT", "LOW_CONFIDENCE"]
BlockState = Literal["idle", "active", "warn", "fault", "bypassed"]


@dataclass
class Channel:
    """A block of time-domain samples from one signal tap."""
    rate_hz: float
    samples: List[float]
    unit: str = "norm"
    clipped: bool = False            # did this tap hit its AOP / full scale?
    label: str = ""


@dataclass
class Spectrum:
    f0_hz: float
    df_hz: float
    mag_db: List[float]
    label: str = ""


@dataclass
class Scalar:
    """One number, with everything the UI needs to render it without knowing
    what it means: unit, expected range, and whether it is currently good."""
    value: Optional[float]
    unit: str = ""
    lo: Optional[float] = None
    hi: Optional[float] = None
    good: Optional[bool] = None      # None = no opinion
    group: str = "misc"              # tiles are grouped by this
    fmt: str = ".2f"
    label: str = ""


@dataclass
class Vector:
    """An array to plot as a curve -- filter coefficients, a magnitude response,
    a class posterior."""
    values: List[float]
    x0: float = 0.0
    dx: float = 1.0
    unit: str = ""
    label: str = ""
    kind: str = "line"               # line | bar | stem


@dataclass
class Block:
    """A node in the signal-flow diagram."""
    state: BlockState = "idle"
    load: Optional[float] = None     # 0..1, fraction of its deadline consumed
    latency_us: Optional[float] = None
    rate_hz: Optional[float] = None
    detail: str = ""


@dataclass
class Edge:
    """A connection in the flow diagram. `level` drives the animation."""
    src: str
    dst: str
    level: Optional[float] = None    # 0..1 signal activity
    label: str = ""


@dataclass
class Geometry:
    """Head-and-source view."""
    head_yaw_deg: float = 0.0
    source_az_deg: Optional[float] = None       # true, if known (sim only)
    est_az_deg: Optional[float] = None          # what the pipeline believes
    conf: Optional[float] = None
    causal_L: Optional[bool] = None
    causal_R: Optional[bool] = None
    active_ref: Optional[str] = None            # "L" | "R" | None


@dataclass
class Event:
    kind: str
    t: float
    severity: str = "info"           # info | warn | alarm
    text: str = ""


@dataclass
class TelemetryFrame:
    t: float                                     # seconds since producer start
    seq: int
    source: SourceKind = "sim"
    experiment: str = ""
    mode: SystemMode = "NORMAL"
    schema: str = SCHEMA_VERSION
    wall: float = field(default_factory=time.time)
    channels: Dict[str, Channel] = field(default_factory=dict)
    spectra: Dict[str, Spectrum] = field(default_factory=dict)
    scalars: Dict[str, Scalar] = field(default_factory=dict)
    vectors: Dict[str, Vector] = field(default_factory=dict)
    blocks: Dict[str, Block] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    geometry: Optional[Geometry] = None
    events: List[Event] = field(default_factory=list)
    notes: str = ""

    def to_dict(self):
        return asdict(self)


# The canonical signal-flow graph. Producers report state for the blocks they
# actually run; the UI greys out any block that no producer reports, which is how
# a laptop-only run and a full headset run share one diagram.
FLOW_NODES = [
    ("mic_ref_l",  "Ref mic L",      0.06, 0.20),
    ("mic_ref_r",  "Ref mic R",      0.06, 0.44),
    ("mic_err",    "Error mic",      0.06, 0.68),
    ("mic_boom",   "Boom mic",       0.06, 0.88),
    ("afe",        "Analog front end\n+ limiter", 0.24, 0.40),
    ("codec",      "ANC codec\nADC/DAC",          0.40, 0.40),
    ("l0",         "L0 control\n48 kHz FxNLMS",   0.58, 0.40),
    ("l2",         "L2 scene engine\n62.5 Hz",    0.58, 0.12),
    ("l1",         "L1 speech\nenhancer",         0.58, 0.86),
    ("driver",     "Driver",         0.80, 0.40),
    ("ear",        "Ear",            0.94, 0.40),
    ("radio",      "Radio / PTT",    0.80, 0.86),
]

FLOW_EDGES = [
    ("mic_ref_l", "afe"), ("mic_ref_r", "afe"), ("mic_err", "afe"),
    ("afe", "codec"), ("codec", "l0"), ("codec", "l2"),
    ("l2", "l0"),                      # the coefficient path -- not audio
    ("l0", "driver"), ("driver", "ear"), ("ear", "mic_err"),
    ("mic_boom", "l1"), ("l1", "radio"), ("l2", "l1"),
]

# Edges that carry coefficients rather than audio. The UI draws these differently
# because the distinction is the whole architecture.
COEFFICIENT_EDGES = {("l2", "l0"), ("l2", "l1")}
