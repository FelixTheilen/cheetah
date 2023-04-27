import cheetah
import numpy as np
import torch

ParameterBeam_parameters = cheetah.ParameterBeam.from_parameters(sigma_x=175e-9, sigma_y=175e-9)
ParticleBeam_parameters = cheetah.ParticleBeam.from_parameters(sigma_x=175e-9, sigma_y=175e-9)

ParameterBeam_astra = cheetah.ParameterBeam.from_astra(
    "../benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)
ParticleBeam_astra = cheetah.ParticleBeam.from_astra(
    "../benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)

#Expected Results
ParticleBeam_n = 100000
ParameterBeam_Energy = 107315902.44355084
ParticleBeam_Energy = 107315902.44355084
ParameterBeam_mu = torch.tensor(
    [
        8.2413e-07,
        5.9885e-08,
        -1.7276e-06,
        -1.1746e-07,
        5.7250e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
ParticleBeam_mu = torch.tensor(
    [
        8.2413e-07,
        5.9885e-08,
        -1.7276e-06,
        -1.1746e-07,
        5.7250e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
ParameterBeam_cov = torch.tensor(
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
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
ParticleBeam_cov = [
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

def test_ParticleBeam_n():
    assert ParticleBeam.n == 100000


def test_ParameterBeam_energy():
    actual = ParameterBeam.energy
    expected = 100000000.000
    assert np.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_ParticleBeam_energy():
    actual = ParticleBeam.energy
    expected = 100000000.000
    assert np.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_ParameterBeam_mu():
    actual = ParameterBeam._mu
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_ParticleBeam_mean():
    actual = ParticleBeam.particles.mean(axis=0)
    expected = torch.tensor(
        [
            -9.6960e-11,
            -5.3534e-11,
            4.4916e-11,
            9.2226e-10,
            5.1672e-09,
            1.9966e-09,
            1.0000e00,
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


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
                3.0625e-14,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                4.0000e-14,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                3.0625e-14,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                4.0000e-14,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                1.0000e-12,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                1.0000e-12,
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
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_ParticleBeam_cov():
    actual = np.cov(ParticleBeam.particles.t().numpy())
    expected = torch.tensor(
        [
            [
                3.07948938e-14,
                -1.83413905e-16,
                -1.68399788e-16,
                4.25610929e-17,
                -7.98809553e-16,
                4.54601961e-16,
                0.00000000e00,
            ],
            [
                -1.83413905e-16,
                3.99321871e-14,
                1.52709870e-16,
                5.61528738e-19,
                9.21760507e-17,
                2.64697265e-16,
                0.00000000e00,
            ],
            [
                -1.68399788e-16,
                1.52709870e-16,
                3.05111147e-14,
                -1.70158009e-16,
                1.68751952e-16,
                -4.03071599e-16,
                0.00000000e00,
            ],
            [
                4.25610929e-17,
                5.61528738e-19,
                -1.70158009e-16,
                4.02518627e-14,
                -6.30815231e-17,
                2.41067632e-17,
                0.00000000e00,
            ],
            [
                -7.98809553e-16,
                9.21760507e-17,
                1.68751952e-16,
                -6.30815231e-17,
                9.89274528e-13,
                1.88611486e-15,
                0.00000000e00,
            ],
            [
                4.54601961e-16,
                2.64697265e-16,
                -4.03071599e-16,
                2.41067632e-17,
                1.88611486e-15,
                9.99455081e-13,
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
    assert np.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)

def test_ParameterBeam_ParticleBeam_cov_dif():
    assert np.allclose(
        ParameterBeam._cov,
        np.cov(ParticleBeam.particles.t().numpy()),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


ParameterBeam_transformed_Sigma = ParameterBeam.transformed_to(sigma_x=123e-6)
ParticleBeam_transformed_Sigma = ParticleBeam.transformed_to(sigma_x=123e-6)


def test_ParticleBeam_transformed_Sigma_n():
    assert ParticleBeam_transformed_Sigma.n == 100000


def test_ParameterBeam_transformed_Sigma_energy():
    actual = ParameterBeam_transformed_Sigma.energy
    expected = 100000000.000
    assert np.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_ParticleBeam_transformed_Sigma_energy():
    actual = ParticleBeam_transformed_Sigma.energy
    expected = 100000000.000
    assert np.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_ParameterBeam_transformed_Sigma_mu():
    actual = ParameterBeam_transformed_Sigma._mu
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_ParticleBeam_transformed_Sigma_mean():
    actual = ParticleBeam_transformed_Sigma.particles.mean(axis=0)
    expected = torch.tensor(
        [
            -9.6960e-11,
            -5.3534e-11,
            4.4916e-11,
            9.2226e-10,
            5.1672e-09,
            1.9966e-09,
            1.0000e00,
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_ParameterBeam_transformed_Sigma_ParticleBeam_transformed_Sigma_mu_dif():
    assert torch.allclose(
        ParameterBeam_transformed_Sigma._mu,
        ParticleBeam_transformed_Sigma.particles.mean(axis=0),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_ParameterBeam_transformed_Sigma_cov():
    actual = ParameterBeam_transformed_Sigma._cov
    expected = torch.tensor(
        [
            [
                1.5129e-08,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
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
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
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
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                1.0000e-12,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                1.0000e-12,
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
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_ParticleBeam_transformed_Sigma_cov():
    actual = np.cov(ParticleBeam_transformed_Sigma.particles.t().numpy())
    expected = torch.tensor(
        [
            [
                3.07948938e-14,
                -1.83413905e-16,
                -1.68399788e-16,
                4.25610929e-17,
                -7.98809553e-16,
                4.54601961e-16,
                0.00000000e00,
            ],
            [
                -1.83413905e-16,
                3.99321871e-14,
                1.52709870e-16,
                5.61528738e-19,
                9.21760507e-17,
                2.64697265e-16,
                0.00000000e00,
            ],
            [
                -1.68399788e-16,
                1.52709870e-16,
                3.05111147e-14,
                -1.70158009e-16,
                1.68751952e-16,
                -4.03071599e-16,
                0.00000000e00,
            ],
            [
                4.25610929e-17,
                5.61528738e-19,
                -1.70158009e-16,
                4.02518627e-14,
                -6.30815231e-17,
                2.41067632e-17,
                0.00000000e00,
            ],
            [
                -7.98809553e-16,
                9.21760507e-17,
                1.68751952e-16,
                -6.30815231e-17,
                9.89274528e-13,
                1.88611486e-15,
                0.00000000e00,
            ],
            [
                4.54601961e-16,
                2.64697265e-16,
                -4.03071599e-16,
                2.41067632e-17,
                1.88611486e-15,
                9.99455081e-13,
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
    assert np.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_ParameterBeam_transformed_Sigma_ParticleBeam_transformed_Sigma_cov_dif():
    assert np.allclose(
        ParameterBeam_transformed_Sigma._cov,
        np.cov(ParticleBeam_transformed_Sigma.particles.t().numpy()),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )
