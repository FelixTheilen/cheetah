import ARESlatticeStage3v1_9 as ares
import numpy as np
import torch
from cheetah import (
    ParameterBeam,
    ParticleBeam,
    Segment
)

"""
Test Beam, which can be found in GitHub in the folder
benchmark/cheetah/ACHIP_EA1_2021.1351.001
"""


ParameterBeam = ParameterBeam.from_astra("../benchmark/cheetah/ACHIP_EA1_2021.1351.001")
ParticleBeam = ParticleBeam.from_astra("../benchmark/cheetah/ACHIP_EA1_2021.1351.001")


def test_ParticleBeam_n():
    assert ParticleBeam.n == 100000


def test_ParameterBeam_energy():
    actual = ParameterBeam.energy
    expected = 107315902.44355084
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParticleBeam_energy():
    actual = ParticleBeam.energy
    expected = 107315902.44355084
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParameterBeam_mu():
    actual = ParameterBeam._mu
    expected = torch.tensor(
        [
            8.2413e-07,
            5.9885e-08,
            -1.7276e-06,
            -1.1746e-07,
            5.7250e-06,
            3.8292e-04,
            1.0000e00,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParticleBeam_particles_mean():
    actual = ParticleBeam.particles.mean(axis=0)
    expected = torch.tensor(
        [
            8.2413e-07,
            5.9885e-08,
            -1.7276e-06,
            -1.1746e-07,
            5.7250e-06,
            3.8292e-04,
            1.0000e00,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParameterBeam_ParticleBeam_mu_dif():
    assert torch.allclose(
        ParameterBeam._mu,
        ParticleBeam.particles.mean(axis=0),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_ParameterBeam_cov():
    actual = ParameterBeam._cov
    expected = torch.tensor(
        [
            [
                3.0589e-08,
                5.5679e-10,
                8.0207e-13,
                2.2111e-13,
                -6.3953e-13,
                -7.6916e-12,
                0.0000e00,
            ],
            [
                5.5679e-10,
                1.3538e-11,
                9.8643e-14,
                6.4855e-15,
                -3.6896e-14,
                -8.0708e-14,
                0.0000e00,
            ],
            [
                8.0207e-13,
                9.8643e-14,
                3.0693e-08,
                5.6076e-10,
                6.0425e-13,
                3.3948e-11,
                0.0000e00,
            ],
            [
                2.2111e-13,
                6.4855e-15,
                5.6076e-10,
                1.3646e-11,
                6.4452e-14,
                5.3652e-12,
                0.0000e00,
            ],
            [
                -6.3953e-13,
                -3.6896e-14,
                6.0425e-13,
                6.4452e-14,
                6.4185e-11,
                3.0040e-09,
                0.0000e00,
            ],
            [
                -7.6916e-12,
                -8.0708e-14,
                3.3948e-11,
                5.3652e-12,
                3.0040e-09,
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


def test_ParticleBeam_cov():
    actual = np.cov(ParticleBeam.particles.t().numpy())
    expected = np.array(
        [
            [
                3.05892720e-08,
                5.56790623e-10,
                8.02068059e-13,
                2.21114027e-13,
                -6.39527178e-13,
                -7.69157365e-12,
                0.00000000e00,
            ],
            [
                5.56790623e-10,
                1.35380005e-11,
                9.86434299e-14,
                6.48545528e-15,
                -3.68956168e-14,
                -8.07087941e-14,
                0.00000000e00,
            ],
            [
                8.02068059e-13,
                9.86434299e-14,
                3.06934414e-08,
                5.60758572e-10,
                6.04253726e-13,
                3.39481957e-11,
                0.00000000e00,
            ],
            [
                2.21114027e-13,
                6.48545528e-15,
                5.60758572e-10,
                1.36463749e-11,
                6.44515124e-14,
                5.36516751e-12,
                0.00000000e00,
            ],
            [
                -6.39527178e-13,
                -3.68956168e-14,
                6.04253726e-13,
                6.44515124e-14,
                6.41849709e-11,
                3.00400242e-09,
                0.00000000e00,
            ],
            [
                -7.69157365e-12,
                -8.07087941e-14,
                3.39481957e-11,
                5.36516751e-12,
                3.00400242e-09,
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
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParameterBeam_ParticleBeam_cov_dif():
    assert np.allclose(
        ParameterBeam._cov,
        np.cov(ParticleBeam.particles.t().numpy()),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


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
