import ARESlatticeStage3v1_9 as lattice
import cheetah
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from cheetah import accelerator, particles

beam3 = cheetah.ParticleBeam.from_parameters(sigma_x=175e-9, sigma_y=175e-9)

print("beam3.particles.mean(axis=0) = ")
print(beam3.particles.mean(axis=0))

print("np.cov(beam3.particles.t().numpy()) = ")
print(np.cov(beam3.particles.t().numpy()))


"""
cell = cheetah.utils.subcell_of_ocelot(lattice.cell, "AREASOLA1", "ARMRBSCR1")
segment = cheetah.Segment.from_ocelot(cell)
segment.ARMRBSCR1.is_active = True

segment.AREAMQZM2.misalignment = (0.0000005, 0.0)

segment.AREAMQZM1.k1 = 4.5
segment.AREAMQZM2.k1 = -9.0
segment.AREAMQZM3.k1 = 4.5

segment.plot_overview(resolution=0.01)

matplotlib.pyplot.savefig("reference_plot.png")
"""

"""
i = 0.18
n = 20
x = 0
y = 0

segment = accelerator.Segment([
    accelerator.Quadrupole(length=i, k1 = n, misalignment = (x,y), name="Quadrupole")
])
incoming_beam = particles.ParameterBeam.from_parameters()

outgoing_beam = segment(incoming_beam)

print(outgoing_beam)

"""

"""
for i in range(1, 10):
    for n in range(1, 10):
        for x in range(1, 10):
            for y in range(1, 10):
                segment = accelerator.Segment([
                    accelerator.Quadrupole(length=i, k1 = n, misalignment = (x,y), name="Quadrupole")
                ])
                incoming_beam = particles.ParameterBeam.from_parameters()

                outgoing_beam = segment(incoming_beam)

                print(outgoing_beam)
"""
