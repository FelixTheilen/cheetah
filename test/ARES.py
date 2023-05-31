import matplotlib.pyplot as plt
from cheetah.accelerator import (
    Drift,
    Quadrupole,
    Segment,
)
from cheetah.particles import ParameterBeam

segment = Segment(
    [
        Drift(length=0.2, name="Drift_AREASOLA1"),
        Quadrupole(length=0.12, k1=-40, misalignment=(0.01, 0), name="AREAMQZM1"),
        Drift(length=0.2, name="Drift_AREAMQZM1"),
    ]
)


incoming_beam = ParameterBeam.from_parameters()

outgoing_beam = segment(incoming_beam)

segment.plot_overview(beam=outgoing_beam)

plt.show()
