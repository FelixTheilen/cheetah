import ARESlatticeStage3v1_9 as ares
import numpy as np
import torch
from cheetah import ParameterBeam, ParticleBeam, Segment

"""
Test Beam, which can be found in GitHub in the folder
benchmark/cheetah/ACHIP_EA1_2021.1351.001
"""


ParameterBeam = ParameterBeam.from_astra("../benchmark/cheetah/ACHIP_EA1_2021.1351.001")
ParticleBeam = ParticleBeam.from_astra("../benchmark/cheetah/ACHIP_EA1_2021.1351.001")


segment = Segment.from_ocelot(ares.cell)

FinalTestResult_ParameterBeam = segment(ParameterBeam)
FinalTestResult_ParticleBeam = segment(ParticleBeam)


def test_FinalTestResult_ParticleBeam_n():
    assert FinalTestResult_ParticleBeam.n == 100000


def test_FinalTestResult_ParameterBeam_energy():
    actual = FinalTestResult_ParameterBeam.energy
    expected = 107315902.44355084
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_FinalTestResult_ParticleBeam_energy():
    actual = FinalTestResult_ParticleBeam.energy
    expected = 107315902.44355084
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_FinalTestResult_ParameterBeam_mu():
    actual = FinalTestResult_ParameterBeam._mu
    expected = torch.tensor(
        [
            3.3602e-06,
            5.9885e-08,
            -6.7022e-06,
            -1.1746e-07,
            6.0613e-06,
            3.8292e-04,
            1.0000e00,
        ]
    )
    assert np.allclose(actual, expected, rtol=1e-3, atol=1e-8, equal_nan=False)


def test_FinalTestResult_ParticleBeam_particles_mean():
    actual = FinalTestResult_ParticleBeam.particles.mean(axis=0)
    expected = torch.tensor(
        [
            3.3602e-06,
            5.9885e-08,
            -6.7022e-06,
            -1.1746e-07,
            6.0613e-06,
            3.8292e-04,
            1.0000e00,
        ]
    )
    assert np.allclose(actual, expected, rtol=1e-3, atol=1e-8, equal_nan=False)


def test_FinalTestResult_ParameterBeam_FinalTestResult_ParticleBeam_mu_dif():
    assert torch.allclose(
        FinalTestResult_ParameterBeam._mu,
        FinalTestResult_ParticleBeam.particles.mean(axis=0),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_FinalTestResult_ParameterBeam_cov():
    actual = FinalTestResult_ParameterBeam._cov
    expected = torch.tensor(
        [
            [
                1.0203e-07,
                1.1301e-09,
                2.5975e-11,
                4.9577e-13,
                -2.2118e-12,
                -1.1110e-11,
                0.0000e00,
            ],
            [
                1.1301e-09,
                1.3538e-11,
                3.7330e-13,
                6.4855e-15,
                -3.6967e-14,
                -8.0708e-14,
                0.0000e00,
            ],
            [
                2.5975e-11,
                3.7330e-13,
                1.0266e-07,
                1.1387e-09,
                3.5631e-12,
                2.6116e-10,
                0.0000e00,
            ],
            [
                4.9577e-13,
                6.4855e-15,
                1.1387e-09,
                1.3646e-11,
                6.9164e-14,
                5.3652e-12,
                0.0000e00,
            ],
            [
                -2.2118e-12,
                -3.6967e-14,
                3.5631e-12,
                6.9164e-14,
                7.3473e-11,
                7.5715e-09,
                0.0000e00,
            ],
            [
                -1.1110e-11,
                -8.0708e-14,
                2.6116e-10,
                5.3652e-12,
                7.5715e-09,
                5.2005e-06,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
        ]
    )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_FinalTestResult_ParticleBeam_cov():
    actual = np.cov(FinalTestResult_ParticleBeam.particles.t().numpy())
    expected = np.array(
        [
            [
                1.02029056e-07,
                1.13011842e-09,
                2.59751889e-11,
                4.95770046e-13,
                -2.21179593e-12,
                -1.11094675e-11,
                0.00000000e00,
            ],
            [
                1.13011842e-09,
                1.35380005e-11,
                3.73299243e-13,
                6.48545528e-15,
                -3.69664988e-14,
                -8.07087941e-14,
                0.00000000e00,
            ],
            [
                2.59751889e-11,
                3.73299243e-13,
                1.02663675e-07,
                1.13867597e-09,
                3.56311933e-12,
                2.61160482e-10,
                0.00000000e00,
            ],
            [
                4.95770046e-13,
                6.48545528e-15,
                1.13867597e-09,
                1.36463749e-11,
                6.91636914e-14,
                5.36516751e-12,
                0.00000000e00,
            ],
            [
                -2.21179593e-12,
                -3.69664988e-14,
                3.56311933e-12,
                6.91636914e-14,
                7.34733682e-11,
                7.57152810e-09,
                0.00000000e00,
            ],
            [
                -1.11094675e-11,
                -8.07087941e-14,
                2.61160482e-10,
                5.36516751e-12,
                7.57152810e-09,
                5.20046739e-06,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
        ]
    )
    assert np.allclose(actual, expected, rtol=1e-05, atol=1e-14, equal_nan=False)


def test_FinalTestResult_ParameterBeam_FinalTestResult_ParticleBeam_cov_dif():
    assert np.allclose(
        FinalTestResult_ParameterBeam._cov,
        np.cov(FinalTestResult_ParticleBeam.particles.t().numpy()),
        rtol=1e-3,
        atol=1e-8,
        equal_nan=True,
    )
