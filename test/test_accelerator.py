import cheetah
import numpy as np
import torch

"""
Test Beam, which can be found in GitHub in the folder
benchmark/cheetah/ACHIP_EA1_2021.1351.001
"""


ParameterBeam = cheetah.ParameterBeam.from_astra(
    "../benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)
ParticleBeam = cheetah.ParticleBeam.from_astra(
    "../benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)

#Segments
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

Screen = cheetah.Segment(
    [cheetah.Screen(resolution=(1000, 1000), pixel_size=1, name="element")]
)

Undulator = cheetah.Segment([cheetah.Undulator(length=0.02, name="element")])

#Segments applied on Beams
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
HorizontalCorrector_ParticleBeam_n = 100000
HorizontalCorrector_ParameterBeam_Energy = 107315902.44355084
HorizontalCorrector_ParticleBeam_Energy = 107315902.44355084
HorizontalCorrector_ParameterBeam_mu = torch.tensor(
    [
        8.2532e-07,
        2.0001e-03,
        -1.7300e-06,
        -1.1746e-07,
        5.7250e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
HorizontalCorrector_ParticleBeam_mu = torch.tensor(
    [
        8.2532e-07,
        2.0001e-03,
        -1.7300e-06,
        -1.1746e-07,
        5.7250e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
HorizontalCorrector_ParameterBeam_cov = torch.tensor(
    [
        [
            3.0612e-08,
            5.5706e-10,
            8.0846e-13,
            2.2124e-13,
            -6.4027e-13,
            -7.6932e-12,
            0.0000e00,
        ],
        [
            5.5706e-10,
            1.3538e-11,
            9.8773e-14,
            6.4855e-15,
            -3.6896e-14,
            -8.0708e-14,
            0.0000e00,
        ],
        [
            8.0846e-13,
            9.8773e-14,
            3.0716e-08,
            5.6103e-10,
            6.0554e-13,
            3.4055e-11,
            0.0000e00,
        ],
        [
            2.2124e-13,
            6.4855e-15,
            5.6103e-10,
            1.3646e-11,
            6.4452e-14,
            5.3652e-12,
            0.0000e00,
        ],
        [
            -6.4027e-13,
            -3.6896e-14,
            6.0554e-13,
            6.4452e-14,
            6.4185e-11,
            3.0040e-09,
            0.0000e00,
        ],
        [
            -7.6932e-12,
            -8.0708e-14,
            3.4055e-11,
            5.3652e-12,
            3.0040e-09,
            5.2005e-06,
            0.0000e00,
        ],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
HorizontalCorrector_ParticleBeam_cov = [
    [
        3.06115490e-08,
        5.57061327e-10,
        8.08462999e-13,
        2.21243689e-13,
        -6.40264932e-13,
        -7.69321934e-12,
        0.00000000e00,
    ],
    [
        5.57061327e-10,
        1.35379983e-11,
        9.87493703e-14,
        6.48437255e-15,
        -3.68937077e-14,
        -8.13532478e-14,
        0.00000000e00,
    ],
    [
        8.08462999e-13,
        9.87493703e-14,
        3.07158772e-08,
        5.61031499e-10,
        6.05542772e-13,
        3.40555441e-11,
        0.00000000e00,
    ],
    [
        2.21243689e-13,
        6.48437255e-15,
        5.61031499e-10,
        1.36463749e-11,
        6.44515124e-14,
        5.36516751e-12,
        0.00000000e00,
    ],
    [
        -6.40264932e-13,
        -3.68937077e-14,
        6.05542772e-13,
        6.44515124e-14,
        6.41849709e-11,
        3.00400242e-09,
        0.00000000e00,
    ],
    [
        -7.69321934e-12,
        -8.13532478e-14,
        3.40555441e-11,
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
VerticalCorrector_ParticleBeam_n = 100000
VerticalCorrector_ParameterBeam_Energy = 107315902.44355084
VerticalCorrector_ParticleBeam_Energy = 107315902.44355084
VerticalCorrector_ParameterBeam_mu = torch.tensor(
    [8.2532e-07, 5.9885e-08, -1.7300e-06, 1.9999e-03, 5.7250e-06, 3.8292e-04, 1.0000e00]
)
VerticalCorrector_ParticleBeam_mu = torch.tensor(
    [8.2532e-07, 5.9885e-08, -1.7300e-06, 1.9999e-03, 5.7250e-06, 3.8292e-04, 1.0000e00]
)
VerticalCorrector_ParameterBeam_cov = torch.tensor(
    [
        [
            3.0612e-08,
            5.5706e-10,
            8.0846e-13,
            2.2124e-13,
            -6.4027e-13,
            -7.6932e-12,
            0.0000e00,
        ],
        [
            5.5706e-10,
            1.3538e-11,
            9.8773e-14,
            6.4855e-15,
            -3.6896e-14,
            -8.0708e-14,
            0.0000e00,
        ],
        [
            8.0846e-13,
            9.8773e-14,
            3.0716e-08,
            5.6103e-10,
            6.0554e-13,
            3.4055e-11,
            0.0000e00,
        ],
        [
            2.2124e-13,
            6.4855e-15,
            5.6103e-10,
            1.3646e-11,
            6.4452e-14,
            5.3652e-12,
            0.0000e00,
        ],
        [
            -6.4027e-13,
            -3.6896e-14,
            6.0554e-13,
            6.4452e-14,
            6.4185e-11,
            3.0040e-09,
            0.0000e00,
        ],
        [
            -7.6932e-12,
            -8.0708e-14,
            3.4055e-11,
            5.3652e-12,
            3.0040e-09,
            5.2005e-06,
            0.0000e00,
        ],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
VerticalCorrector_ParticleBeam_cov = [
    [
        3.06115490e-08,
        5.57061383e-10,
        8.08462999e-13,
        2.21235109e-13,
        -6.40264932e-13,
        -7.69321934e-12,
        0.00000000e00,
    ],
    [
        5.57061383e-10,
        1.35380005e-11,
        9.87730674e-14,
        6.48450760e-15,
        -3.68956168e-14,
        -8.07087941e-14,
        0.00000000e00,
    ],
    [
        8.08462999e-13,
        9.87730674e-14,
        3.07158772e-08,
        5.61031529e-10,
        6.05542772e-13,
        3.40555441e-11,
        0.00000000e00,
    ],
    [
        2.21235109e-13,
        6.48450760e-15,
        5.61031529e-10,
        1.36463746e-11,
        6.44517278e-14,
        5.36540115e-12,
        0.00000000e00,
    ],
    [
        -6.40264932e-13,
        -3.68956168e-14,
        6.05542772e-13,
        6.44517278e-14,
        6.41849709e-11,
        3.00400242e-09,
        0.00000000e00,
    ],
    [
        -7.69321934e-12,
        -8.07087941e-14,
        3.40555441e-11,
        5.36540115e-12,
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
Cavity_ParticleBeam_n = 100000
Cavity_ParameterBeam_Energy = 107316902.44355084
Cavity_ParticleBeam_Energy = 107316902.44355084
Cavity_ParameterBeam_mu = torch.tensor(
    [
        8.2532e-07,
        5.9885e-08,
        -1.7300e-06,
        -1.1746e-07,
        5.7252e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
Cavity_ParticleBeam_mu = torch.tensor(
    [
        8.2532e-07,
        5.9885e-08,
        -1.7300e-06,
        -1.1746e-07,
        5.7252e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
Cavity_ParameterBeam_cov = torch.tensor(
    [
        [
            3.0612e-08,
            5.5706e-10,
            8.0846e-13,
            2.2124e-13,
            -6.4027e-13,
            -7.6932e-12,
            0.0000e00,
        ],
        [
            5.5706e-10,
            1.3538e-11,
            9.8773e-14,
            6.4855e-15,
            -3.6896e-14,
            -8.0708e-14,
            0.0000e00,
        ],
        [
            8.0846e-13,
            9.8773e-14,
            3.0716e-08,
            5.6103e-10,
            6.0556e-13,
            3.4055e-11,
            0.0000e00,
        ],
        [
            2.2124e-13,
            6.4855e-15,
            5.6103e-10,
            1.3646e-11,
            6.4454e-14,
            5.3652e-12,
            0.0000e00,
        ],
        [
            -6.4027e-13,
            -3.6896e-14,
            6.0556e-13,
            6.4454e-14,
            6.4188e-11,
            3.0064e-09,
            0.0000e00,
        ],
        [
            -7.6932e-12,
            -8.0708e-14,
            3.4055e-11,
            5.3652e-12,
            3.0064e-09,
            5.2005e-06,
            0.0000e00,
        ],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
Cavity_ParticleBeam_cov = [
    [
        3.06115490e-08,
        5.57061383e-10,
        8.08462999e-13,
        2.21243689e-13,
        -6.40268513e-13,
        -7.69321934e-12,
        0.00000000e00,
    ],
    [
        5.57061383e-10,
        1.35380005e-11,
        9.87730674e-14,
        6.48545528e-15,
        -3.68956562e-14,
        -8.07087941e-14,
        0.00000000e00,
    ],
    [
        8.08462999e-13,
        9.87730674e-14,
        3.07158772e-08,
        5.61031499e-10,
        6.05558103e-13,
        3.40555441e-11,
        0.00000000e00,
    ],
    [
        2.21243689e-13,
        6.48545528e-15,
        5.61031499e-10,
        1.36463749e-11,
        6.44539428e-14,
        5.36516751e-12,
        0.00000000e00,
    ],
    [
        -6.40268513e-13,
        -3.68956562e-14,
        6.05558103e-13,
        6.44539428e-14,
        6.41876963e-11,
        3.00636064e-09,
        0.00000000e00,
    ],
    [
        -7.69321934e-12,
        -8.07087941e-14,
        3.40555441e-11,
        5.36516751e-12,
        3.00636064e-09,
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
BPM_ParticleBeam_n = 100000
BPM_ParameterBeam_Energy = 107315902.44355084
BPM_ParticleBeam_Energy = 107315902.44355084
BPM_ParameterBeam_mu = torch.tensor(
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
BPM_ParticleBeam_mu = torch.tensor(
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
BPM_ParameterBeam_cov = torch.tensor(
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
BPM_ParticleBeam_cov = [
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
Screen_ParticleBeam_n = 100000
Screen_ParameterBeam_Energy = 107315902.44355084
Screen_ParticleBeam_Energy = 107315902.44355084
Screen_ParameterBeam_mu = torch.tensor(
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
Screen_ParticleBeam_mu = torch.tensor(
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
Screen_ParameterBeam_cov = torch.tensor(
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
Screen_ParticleBeam_cov = [
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
Undulator_ParticleBeam_n = 100000
Undulator_ParameterBeam_Energy = 107315902.44355084
Undulator_ParticleBeam_Energy = 107315902.44355084
Undulator_ParameterBeam_mu = torch.tensor(
    [
        8.2532e-07,
        5.9885e-08,
        -1.7300e-06,
        -1.1746e-07,
        5.7252e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
Undulator_ParticleBeam_mu = torch.tensor(
    [
        8.2532e-07,
        5.9885e-08,
        -1.7300e-06,
        -1.1746e-07,
        5.7252e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
Undulator_ParameterBeam_cov = torch.tensor(
    [
        [
            3.0612e-08,
            5.5706e-10,
            8.0846e-13,
            2.2124e-13,
            -6.4027e-13,
            -7.6932e-12,
            0.0000e00,
        ],
        [
            5.5706e-10,
            1.3538e-11,
            9.8773e-14,
            6.4855e-15,
            -3.6896e-14,
            -8.0708e-14,
            0.0000e00,
        ],
        [
            8.0846e-13,
            9.8773e-14,
            3.0716e-08,
            5.6103e-10,
            6.0556e-13,
            3.4055e-11,
            0.0000e00,
        ],
        [
            2.2124e-13,
            6.4855e-15,
            5.6103e-10,
            1.3646e-11,
            6.4454e-14,
            5.3652e-12,
            0.0000e00,
        ],
        [
            -6.4027e-13,
            -3.6896e-14,
            6.0556e-13,
            6.4454e-14,
            6.4188e-11,
            3.0064e-09,
            0.0000e00,
        ],
        [
            -7.6932e-12,
            -8.0708e-14,
            3.4055e-11,
            5.3652e-12,
            3.0064e-09,
            5.2005e-06,
            0.0000e00,
        ],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
Undulator_ParticleBeam_cov = [
    [
        3.06115490e-08,
        5.57061383e-10,
        8.08462999e-13,
        2.21243689e-13,
        -6.40268513e-13,
        -7.69321934e-12,
        0.00000000e00,
    ],
    [
        5.57061383e-10,
        1.35380005e-11,
        9.87730674e-14,
        6.48545528e-15,
        -3.68956562e-14,
        -8.07087941e-14,
        0.00000000e00,
    ],
    [
        8.08462999e-13,
        9.87730674e-14,
        3.07158772e-08,
        5.61031499e-10,
        6.05558103e-13,
        3.40555441e-11,
        0.00000000e00,
    ],
    [
        2.21243689e-13,
        6.48545528e-15,
        5.61031499e-10,
        1.36463749e-11,
        6.44539428e-14,
        5.36516751e-12,
        0.00000000e00,
    ],
    [
        -6.40268513e-13,
        -3.68956562e-14,
        6.05558103e-13,
        6.44539428e-14,
        6.41876963e-11,
        3.00636064e-09,
        0.00000000e00,
    ],
    [
        -7.69321934e-12,
        -8.07087941e-14,
        3.40555441e-11,
        5.36516751e-12,
        3.00636064e-09,
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

#Tests
def test_ParticleBeam_n():
    assert ParticleBeam.n == ParticleBeam_n


def test_ParameterBeam_Energy():
    actual = ParameterBeam.energy
    assert np.isclose(
        actual, ParameterBeam_Energy, rtol=1e-4, atol=1e-8, equal_nan=False
    )


def test_ParticleBeam_Energy():
    actual = ParticleBeam.energy
    assert np.isclose(
        actual, ParticleBeam_Energy, rtol=1e-4, atol=1e-8, equal_nan=False
    )


def test_ParameterBeam_mu():
    actual = ParameterBeam._mu
    assert torch.allclose(
        actual, ParameterBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_ParticleBeam_mu():
    actual = ParticleBeam.particles.mean(axis=0)
    assert torch.allclose(
        actual, ParticleBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_ParameterBeam_ParticleBeam_mu_dif():
    assert torch.allclose(
        ParameterBeam._mu,
        ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_ParameterBeam_cov():
    actual = ParameterBeam._cov
    assert torch.allclose(
        actual, ParameterBeam_cov, rtol=1e-4, atol=1e-16, equal_nan=False
    )


def test_ParticleBeam_cov():
    actual = np.cov(ParticleBeam.particles.t().numpy())
    assert np.allclose(actual, ParticleBeam_cov, rtol=1e-4, atol=1e-16, equal_nan=False)


def test_ParameterBeam_ParticleBeam_cov_dif():
    assert np.allclose(
        ParameterBeam._cov,
        np.cov(ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


# HorizontalCorrector
def test_HorizontalCorrector_ParticleBeam_n():
    assert HorizontalCorrector_ParticleBeam.n == HorizontalCorrector_ParticleBeam_n


def test_HorizontalCorrector_ParameterBeam_Energy():
    actual = HorizontalCorrector_ParameterBeam.energy
    assert np.isclose(
        actual,
        HorizontalCorrector_ParameterBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParticleBeam_Energy():
    actual = HorizontalCorrector_ParticleBeam.energy
    assert np.isclose(
        actual,
        HorizontalCorrector_ParticleBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParameterBeam_mu():
    actual = HorizontalCorrector_ParameterBeam._mu
    assert torch.allclose(
        actual,
        HorizontalCorrector_ParameterBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParticleBeam_mu():
    actual = HorizontalCorrector_ParticleBeam.particles.mean(axis=0)
    assert torch.allclose(
        actual,
        HorizontalCorrector_ParticleBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParameterBeam_HorizontalCorrector_ParticleBeam_mu_dif():
    assert torch.allclose(
        HorizontalCorrector_ParameterBeam._mu,
        HorizontalCorrector_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParameterBeam_cov():
    actual = HorizontalCorrector_ParameterBeam._cov
    assert torch.allclose(
        actual,
        HorizontalCorrector_ParameterBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParticleBeam_cov():
    actual = np.cov(HorizontalCorrector_ParticleBeam.particles.t().numpy())
    assert np.allclose(
        actual,
        HorizontalCorrector_ParticleBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParameterBeam_HorizontalCorrector_ParticleBeam_cov_dif():
    assert np.allclose(
        HorizontalCorrector_ParameterBeam._cov,
        np.cov(HorizontalCorrector_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-15,
        equal_nan=False,
    )


# VerticalCorrector
def test_VerticalCorrector_ParticleBeam_n():
    assert VerticalCorrector_ParticleBeam.n == VerticalCorrector_ParticleBeam_n


def test_VerticalCorrector_ParameterBeam_Energy():
    actual = VerticalCorrector_ParameterBeam.energy
    assert np.isclose(
        actual,
        VerticalCorrector_ParameterBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_VerticalCorrector_ParticleBeam_Energy():
    actual = VerticalCorrector_ParticleBeam.energy
    assert np.isclose(
        actual,
        VerticalCorrector_ParticleBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_VerticalCorrector_ParameterBeam_mu():
    actual = VerticalCorrector_ParameterBeam._mu
    assert torch.allclose(
        actual,
        VerticalCorrector_ParameterBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_VerticalCorrector_ParticleBeam_mu():
    actual = VerticalCorrector_ParticleBeam.particles.mean(axis=0)
    assert torch.allclose(
        actual, VerticalCorrector_ParticleBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_VerticalCorrector_ParameterBeam_VerticalCorrector_ParticleBeam_mu_dif():
    assert torch.allclose(
        VerticalCorrector_ParameterBeam._mu,
        VerticalCorrector_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_VerticalCorrector_ParameterBeam_cov():
    actual = VerticalCorrector_ParameterBeam._cov
    assert torch.allclose(
        actual,
        VerticalCorrector_ParameterBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_VerticalCorrector_ParticleBeam_cov():
    actual = np.cov(VerticalCorrector_ParticleBeam.particles.t().numpy())
    assert np.allclose(
        actual,
        VerticalCorrector_ParticleBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_VerticalCorrector_ParameterBeam_VerticalCorrector_ParticleBeam_cov_dif():
    assert np.allclose(
        VerticalCorrector_ParameterBeam._cov,
        np.cov(VerticalCorrector_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


# Cavity
def test_Cavity_ParticleBeam_n():
    assert Cavity_ParticleBeam.n == Cavity_ParticleBeam_n


def test_Cavity_ParameterBeam_Energy():
    actual = Cavity_ParameterBeam.energy
    assert np.isclose(
        actual, Cavity_ParameterBeam_Energy, rtol=1e-4, atol=1e-8, equal_nan=False
    )


def test_Cavity_ParticleBeam_Energy():
    actual = Cavity_ParticleBeam.energy
    assert np.isclose(
        actual, Cavity_ParticleBeam_Energy, rtol=1e-4, atol=1e-8, equal_nan=False
    )


def test_Cavity_ParameterBeam_mu():
    actual = Cavity_ParameterBeam._mu
    assert torch.allclose(
        actual, Cavity_ParameterBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_Cavity_ParticleBeam_mu():
    actual = Cavity_ParticleBeam.particles.mean(axis=0)
    assert torch.allclose(
        actual, Cavity_ParticleBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_Cavity_ParameterBeam_Cavity_ParticleBeam_mu_dif():
    assert torch.allclose(
        Cavity_ParameterBeam._mu,
        Cavity_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Cavity_ParameterBeam_cov():
    actual = Cavity_ParameterBeam._cov
    assert torch.allclose(
        actual, Cavity_ParameterBeam_cov, rtol=1e-4, atol=1e-16, equal_nan=False
    )


def test_Cavity_ParticleBeam_cov():
    actual = np.cov(Cavity_ParticleBeam.particles.t().numpy())
    assert np.allclose(
        actual, Cavity_ParticleBeam_cov, rtol=1e-4, atol=1e-16, equal_nan=False
    )


def test_Cavity_ParameterBeam_Cavity_ParticleBeam_cov_dif():
    assert np.allclose(
        Cavity_ParameterBeam._cov,
        np.cov(Cavity_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


# BPM
def test_BPM_ParticleBeam_n():
    assert BPM_ParticleBeam.n == BPM_ParticleBeam_n


def test_BPM_ParameterBeam_Energy():
    actual = BPM_ParameterBeam.energy
    assert np.isclose(
        actual, BPM_ParameterBeam_Energy, rtol=1e-4, atol=1e-8, equal_nan=False
    )


def test_BPM_ParticleBeam_Energy():
    actual = BPM_ParticleBeam.energy
    assert np.isclose(
        actual, BPM_ParticleBeam_Energy, rtol=1e-4, atol=1e-8, equal_nan=False
    )


def test_BPM_ParameterBeam_mu():
    actual = BPM_ParameterBeam._mu
    assert torch.allclose(
        actual, BPM_ParameterBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_BPM_ParticleBeam_mu():
    actual = BPM_ParticleBeam.particles.mean(axis=0)
    assert torch.allclose(
        actual, BPM_ParticleBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_BPM_ParameterBeam_BPM_ParticleBeam_mu_dif():
    assert torch.allclose(
        BPM_ParameterBeam._mu,
        BPM_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_BPM_ParameterBeam_cov():
    actual = BPM_ParameterBeam._cov
    assert torch.allclose(
        actual, BPM_ParameterBeam_cov, rtol=1e-4, atol=1e-16, equal_nan=False
    )


def test_BPM_ParticleBeam_cov():
    actual = np.cov(BPM_ParticleBeam.particles.t().numpy())
    assert np.allclose(
        actual, BPM_ParticleBeam_cov, rtol=1e-4, atol=1e-16, equal_nan=False
    )


def test_BPM_ParameterBeam_BPM_ParticleBeam_cov_dif():
    assert np.allclose(
        BPM_ParameterBeam._cov,
        np.cov(BPM_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


# Screen
def test_Screen_ParticleBeam_n():
    assert Screen_ParticleBeam.n == Screen_ParticleBeam_n


def test_Screen_ParameterBeam_Energy():
    actual = Screen_ParameterBeam.energy
    assert np.isclose(
        actual, Screen_ParameterBeam_Energy, rtol=1e-4, atol=1e-8, equal_nan=False
    )


def test_Screen_ParticleBeam_Energy():
    actual = Screen_ParticleBeam.energy
    assert np.isclose(
        actual, Screen_ParticleBeam_Energy, rtol=1e-4, atol=1e-8, equal_nan=False
    )


def test_Screen_ParameterBeam_mu():
    actual = Screen_ParameterBeam._mu
    assert torch.allclose(
        actual, Screen_ParameterBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_Screen_ParticleBeam_mu():
    actual = Screen_ParticleBeam.particles.mean(axis=0)
    assert torch.allclose(
        actual, Screen_ParticleBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_Screen_ParameterBeam_Screen_ParticleBeam_mu_dif():
    assert torch.allclose(
        Screen_ParameterBeam._mu,
        Screen_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Screen_ParameterBeam_cov():
    actual = Screen_ParameterBeam._cov
    assert torch.allclose(
        actual, Screen_ParameterBeam_cov, rtol=1e-4, atol=1e-16, equal_nan=False
    )


def test_Screen_ParticleBeam_cov():
    actual = np.cov(Screen_ParticleBeam.particles.t().numpy())
    assert np.allclose(
        actual, Screen_ParticleBeam_cov, rtol=1e-4, atol=1e-16, equal_nan=False
    )


def test_Screen_ParameterBeam_Screen_ParticleBeam_cov_dif():
    assert np.allclose(
        Screen_ParameterBeam._cov,
        np.cov(Screen_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


# Undulator
def test_Undulator_ParticleBeam_n():
    assert Undulator_ParticleBeam.n == Undulator_ParticleBeam_n


def test_Undulator_ParameterBeam_Energy():
    actual = Undulator_ParameterBeam.energy
    assert np.isclose(
        actual, Undulator_ParameterBeam_Energy, rtol=1e-4, atol=1e-8, equal_nan=False
    )


def test_Undulator_ParticleBeam_Energy():
    actual = Undulator_ParticleBeam.energy
    assert np.isclose(
        actual, Undulator_ParticleBeam_Energy, rtol=1e-4, atol=1e-8, equal_nan=False
    )


def test_Undulator_ParameterBeam_mu():
    actual = Undulator_ParameterBeam._mu
    assert torch.allclose(
        actual, Undulator_ParameterBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_Undulator_ParticleBeam_mu():
    actual = Undulator_ParticleBeam.particles.mean(axis=0)
    assert torch.allclose(
        actual, Undulator_ParticleBeam_mu, rtol=1e-4, atol=1e-9, equal_nan=False
    )


def test_Undulator_ParameterBeam_Undulator_ParticleBeam_mu_dif():
    assert torch.allclose(
        Undulator_ParameterBeam._mu,
        Undulator_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Undulator_ParameterBeam_cov():
    actual = Undulator_ParameterBeam._cov
    assert torch.allclose(
        actual, Undulator_ParameterBeam_cov, rtol=1e-4, atol=1e-16, equal_nan=False
    )


def test_Undulator_ParticleBeam_cov():
    actual = np.cov(Undulator_ParticleBeam.particles.t().numpy())
    assert np.allclose(
        actual, Undulator_ParticleBeam_cov, rtol=1e-4, atol=1e-16, equal_nan=False
    )


def test_Undulator_ParameterBeam_Undulator_ParticleBeam_cov_dif():
    assert np.allclose(
        Undulator_ParameterBeam._cov,
        np.cov(Undulator_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )
