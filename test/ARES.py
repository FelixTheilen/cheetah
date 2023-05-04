from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from cheetah import accelerator, particles, utils
from cheetah.accelerator import (BPM, Drift, HorizontalCorrector, Quadrupole,
                                 Screen, Segment, VerticalCorrector)
from cheetah.particles import Beam, ParameterBeam, ParticleBeam
from cheetah.utils import from_astrabeam
from matplotlib.patches import Rectangle
from scipy import constants
from scipy.stats import multivariate_normal
from torch.distributions import MultivariateNormal

segment = Segment(
    [
        Drift(length=0.00, name="AREASOLA1"),
        Drift(length=0.18, name="Drift_AREASOLA1"),
        Quadrupole(length=0.12, k1=-40, misalignment=(0.01, 0), name="AREAMQZM1"),
        Drift(length=0.43, name="Drift_AREAMQZM1"),
        Quadrupole(length=0.12, k1=-18, misalignment=(0, 0), name="AREAMQZM2"),
        Drift(length=0.20, name="Drift_AREAMQZM2"),
        VerticalCorrector(length=0.02, angle=0.000, name="AREAMCVM1"),
        Drift(length=0.20, name="Drift_AREAMCVM1"),
        Quadrupole(length=0.12, k1=16, misalignment=(0, 0), name="AREAMQZM3"),
        Drift(length=0.18, name="Drift_AREAMQZM3"),
        HorizontalCorrector(length=0.02, angle=0.002, name="AREAMCHM1"),
        Drift(length=0.45, name="Drift_AREAMCHM1"),
        BPM(name="AREABPM1"),
        Screen(
            resolution=(2448, 2040),
            pixel_size=(3.3198e-06, 2.4469e-06),
            binning=1,
            misalignment=(0, 0),
            name="AREABSCR1",
        ),
    ]
)


incoming_beam = ParameterBeam.from_parameters()

outgoing_beam = segment(incoming_beam)

segment.plot_overview(beam=outgoing_beam)

plt.show()
