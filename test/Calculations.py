import cheetah
import numpy as np

ParameterBeam = cheetah.ParameterBeam.from_astra(
    "../benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)
ParticleBeam = cheetah.ParticleBeam.from_astra(
    "../benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)

Drift = cheetah.Segment([cheetah.Drift(length=0.02, name="element")])

HorizontalCorrector = cheetah.Segment(
    [cheetah.HorizontalCorrector(length=0.02, name="element")]
)
HorizontalCorrector.element.angle = 2e-3

VerticalCorrector = cheetah.Segment(
    [cheetah.VerticalCorrector(length=0.02, name="element")]
)
VerticalCorrector.element.angle = 2e-3

Cavity = cheetah.Segment([cheetah.Cavity(length=0.02, name="element")])
Cavity.element.delta_energy = 1000

BPM = cheetah.Segment([cheetah.BPM(name="element")])

Screen = cheetah.Segment([cheetah.Screen(resolution=(1000, 1000),pixel_size=1, name="element")])

Undulator = cheetah.Segment([cheetah.Undulator(length=0.02, name="element")])

HorizontalCorrector_ParameterBeam = HorizontalCorrector(ParameterBeam)
HorizontalCorrector_ParticleBeam = HorizontalCorrector(ParticleBeam)

VerticalCorrector_ParameterBeam = VerticalCorrector(ParameterBeam)
VerticalCorrector_ParticleBeam = VerticalCorrector(ParticleBeam)

Cavity_ParameterBeam = Cavity(ParameterBeam)
Cavity_ParticleBeam = Cavity(ParticleBeam)

BPM_ParameterBeam = BPM(ParameterBeam)
BPM_ParticleBeam = BPM(ParticleBeam)

Screen_ParameterBeam = Screen(ParameterBeam)
Screen_ParticleBeam = Screen(ParticleBeam)

Undulator_ParameterBeam = Undulator(ParameterBeam)
Undulator_ParticleBeam = Undulator(ParticleBeam)

print("ParticleBeam_n = {}".format(ParticleBeam.n))

print("ParameterBeam_Energy = {}".format(ParameterBeam.energy))

print("ParticleBeam_Energy = {}".format(ParticleBeam.energy))

print("ParameterBeam_mu = {}".format(ParameterBeam._mu))

print("ParticleBeam_mu = {}".format(ParticleBeam.particles.mean(axis=0)))

print("ParameterBeam_cov = {}".format(ParameterBeam._cov))

print("ParticleBeam_cov = {}".format(np.cov(ParticleBeam.particles.t().numpy())))

# Horizontal Corrector
print(
    "HorizontalCorrector_ParticleBeam_n = {}".format(HorizontalCorrector_ParticleBeam.n)
)

print(
    "HorizontalCorrector_ParameterBeam_Energy = {}".format(
        HorizontalCorrector_ParameterBeam.energy
    )
)

print(
    "HorizontalCorrector_ParticleBeam_Energy = {}".format(
        HorizontalCorrector_ParticleBeam.energy
    )
)

print(
    "HorizontalCorrector_ParameterBeam_mu = {}".format(
        HorizontalCorrector_ParameterBeam._mu
    )
)

print(
    "HorizontalCorrector_ParticleBeam_mu = {}".format(
        HorizontalCorrector_ParticleBeam.particles.mean(axis=0)
    )
)

print(
    "HorizontalCorrector_ParameterBeam_cov = {}".format(
        HorizontalCorrector_ParameterBeam._cov
    )
)

print(
    "HorizontalCorrector_ParticleBeam_cov = {}".format(
        np.cov(HorizontalCorrector_ParticleBeam.particles.t().numpy())
    )
)

# Vertical Corrector
print("VerticalCorrector_ParticleBeam_n = {}".format(VerticalCorrector_ParticleBeam.n))

print("VerticalCorrector_ParameterBeam_Energy = {}".format(VerticalCorrector_ParameterBeam.energy))

print("VerticalCorrector_ParticleBeam_Energy = {}".format(VerticalCorrector_ParticleBeam.energy))

print("VerticalCorrector_ParameterBeam_mu = {}".format(VerticalCorrector_ParameterBeam._mu))

print("VerticalCorrector_ParticleBeam_mu = {}".format(VerticalCorrector_ParticleBeam.particles.mean(axis=0)))

print("VerticalCorrector_ParameterBeam_cov = {}".format(VerticalCorrector_ParameterBeam._cov))

print("VerticalCorrector_ParticleBeam_cov = {}".format(np.cov(VerticalCorrector_ParticleBeam.particles.t().numpy())))

#Cavity
print("Cavity_ParticleBeam_n = {}".format(Cavity_ParticleBeam.n))

print("Cavity_ParameterBeam_Energy = {}".format(Cavity_ParameterBeam.energy))

print("Cavity_ParticleBeam_Energy = {}".format(Cavity_ParticleBeam.energy))

print("Cavity_ParameterBeam_mu = {}".format(Cavity_ParameterBeam._mu))

print("Cavity_ParticleBeam_mu = {}".format(Cavity_ParticleBeam.particles.mean(axis=0)))

print("Cavity_ParameterBeam_cov = {}".format(Cavity_ParameterBeam._cov))

print("ParticleBeam_cov = {}".format(np.cov(Cavity_ParticleBeam.particles.t().numpy())))

#BPM
print("BPM_ParticleBeam_n = {}".format(BPM_ParticleBeam.n))

print("BPM_ParameterBeam_Energy = {}".format(BPM_ParameterBeam.energy))

print("BPM_ParticleBeam_Energy = {}".format(BPM_ParticleBeam.energy))

print("BPM_ParameterBeam_mu = {}".format(BPM_ParameterBeam._mu))

print("BPM_ParticleBeam_mu = {}".format(BPM_ParticleBeam.particles.mean(axis=0)))

print("BPM_ParameterBeam_cov = {}".format(BPM_ParameterBeam._cov))

print("BPM_ParticleBeam_cov = {}".format(np.cov(BPM_ParticleBeam.particles.t().numpy())))

#Screen
print("Screen_ParticleBeam_n = {}".format(Screen_ParticleBeam.n))

print("Screen_ParameterBeam_Energy = {}".format(Screen_ParameterBeam.energy))

print("Screen_ParticleBeam_Energy = {}".format(Screen_ParticleBeam.energy))

print("Screen_ParameterBeam_mu = {}".format(Screen_ParameterBeam._mu))

print("Screen_ParticleBeam_mu = {}".format(Screen_ParticleBeam.particles.mean(axis=0)))

print("Screen_ParameterBeam_cov = {}".format(Screen_ParameterBeam._cov))

print("Screen_ParticleBeam_cov = {}".format(np.cov(Screen_ParticleBeam.particles.t().numpy())))

#Undulator
print("Undulator_ParticleBeam_n = {}".format(Undulator_ParticleBeam.n))

print("Undulator_ParameterBeam_Energy = {}".format(Undulator_ParameterBeam.energy))

print("Undulator_ParticleBeam_Energy = {}".format(Undulator_ParticleBeam.energy))

print("Undulator_ParameterBeam_mu = {}".format(Undulator_ParameterBeam._mu))

print("Undulator_ParticleBeam_mu = {}".format(Undulator_ParticleBeam.particles.mean(axis=0)))

print("Undulator_ParameterBeam_cov = {}".format(Undulator_ParameterBeam._cov))

print("Undulator_ParticleBeam_cov = {}".format(np.cov(Undulator_ParticleBeam.particles.t().numpy())))
