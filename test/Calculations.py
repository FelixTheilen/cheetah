import sys

import cheetah
import numpy as np
import torch

sys.path.append(
    "c:/users/ftheilen/appdata/local/packages/pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0/localcache/local-packages/python310/site-packages"
)

"""
Test Beam, which can be found in GitHub in the folder benchmark/cheetah/ACHIP_EA1_2021.1351.001
"""
"""
beam1 = cheetah.ParameterBeam.from_astra(
    "D:/Fachpraktikum_DESY/GitHub/cheetah/benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)
beam2 = cheetah.ParticleBeam.from_astra(
    "D:/Fachpraktikum_DESY/GitHub/cheetah/benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)
"""

beam1 = cheetah.ParameterBeam.from_astra(
    "H:/Source/cheetah/benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)
beam2 = cheetah.ParticleBeam.from_astra(
    "H:/Source/cheetah/benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)

segment = cheetah.Segment(
    [cheetah.HorizontalCorrector(length=0.02, name="quad"), cheetah.Drift(length=2.0)]
)
segment.quad.angle = 2e-3

result1 = segment(beam1)
result2 = segment(beam2)

print("ParticleBeam, beam2: n")
print(beam2.n)

print("ParameterBeam, beam1: Energy")
print(beam1.energy)

print("ParticleBeam, beam2: Energy")
print(beam2.energy)

print("ParameterBeam, beam1: cov")
print(beam1._cov)

print("ParticleBeam, beam2: cov")
print(np.cov(beam2.particles.t().numpy()))

print("ParameterBeam, result1: mu")
print(result1._mu)

print("ParticleBeam, result2: mu")
print(result2.particles.mean(axis=0))

#Results
print("ParticleBeam, result2: n")
print(result2.n)

print("ParameterBeam, result1: Energy")
print(result1.energy)

print("ParticleBeam, result2: Energy")
print(result2.energy)
