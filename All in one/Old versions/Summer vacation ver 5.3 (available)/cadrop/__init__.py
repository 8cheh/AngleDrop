"""cadrop - sessile-drop contact angle measurement.

A clean-room reimplementation of the published algorithms behind OpenDrop
(Berry et al., J. Colloid Interface Sci. 454 (2015) 226-237; JOSS 5(51) 2604),
with a few deliberate departures documented in README.md. No OpenDrop source
code is used or derived from -- OpenDrop is GPL-3.0, this is an independent
implementation of the mathematics.
"""
from .geometry import Line
from .measure import MeasureResult, SideFit, measure, measure_file, measure_roi, METHODS
from .detect import detect_baseline, drop_mask, profile_points
from .render import annotate

__version__ = '1.0.0'

__all__ = [
    'Line', 'MeasureResult', 'SideFit', 'measure', 'measure_file', 'measure_roi',
    'METHODS', 'detect_baseline', 'drop_mask', 'profile_points', 'annotate',
]
