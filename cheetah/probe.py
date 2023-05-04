import matplotlib.pyplot as plt
import numpy as np
from cheetah.accelerator import Drift, Segment
from cheetah.particles import Beam, ParameterBeam, ParticleBeam

segment = Segment(
    [
        Drift(length=0.00, name="AREASOLA1"),
        Drift(length=0.18, name="Drift_AREASOLA1"),
    ]
)

incoming_beam = ParameterBeam.from_parameters()

outgoing_beam = segment(incoming_beam)

segment.plot_overview(beam=outgoing_beam)

plt.show()
