# Test with pytest for cheetah
import math

import numpy as np
import torch
from cheetah import accelerator
from scipy import constants

REST_ENERGY = (
    constants.electron_mass
    * constants.speed_of_light**2
    / constants.elementary_charge
)


def test_Drift_length():
    segment = accelerator.Segment(
        [accelerator.Drift(length=0.18, name="Drift_AREASOLA1")]
    )
    assert segment.Drift_AREASOLA1.length == 0.18


def test_Drift_is_active():
    segment = accelerator.Segment(
        [accelerator.Drift(length=0.18, name="Drift_AREASOLA1")]
    )
    assert segment.Drift_AREASOLA1.is_active == True


def test_Drift_is_skippable():
    segment = accelerator.Segment(
        [accelerator.Drift(length=0.18, name="Drift_AREASOLA1")]
    )
    assert segment.Drift_AREASOLA1.is_skippable == True


def test_Drift_transfer_map():
    for i in range(1, 10, 1):
        for j in range(1, 10, 1):

            def round_to(num, dig):
                return round(num, dig - int(math.floor(math.log10(abs(num)))) - 1)

            gamma = j / REST_ENERGY
            igamma2 = 1 / gamma**2 if gamma != 0 else 0
            segment = accelerator.Segment(
                [accelerator.Drift(length=i, name="Drift_AREASOLA1")]
            )
            assert [
                torch.tensor(
                    [
                        [1, i, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, i, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, i * igamma2, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                    ],
                    dtype=torch.float32,
                    device="cpu",
                )
                == segment.Drift_AREASOLA1.transfer_map(energy=j)
            ]


def test_Drift_length():
    segment = accelerator.Segment(
        [accelerator.Drift(length=0.18, name="Drift_AREASOLA1")]
    )
    assert segment.Drift_AREASOLA1.length == 0.18


def test_Drift_is_active():
    segment = accelerator.Segment(
        [accelerator.Drift(length=0.18, name="Drift_AREASOLA1")]
    )
    assert segment.Drift_AREASOLA1.is_active == True


def test_Drift_is_skippable():
    segment = accelerator.Segment(
        [accelerator.Drift(length=0.18, name="Drift_AREASOLA1")]
    )
    assert segment.Drift_AREASOLA1.is_skippable == True


"""
def test_Quadrupole_transfer_map():
    for i in range(0,10,1):
        for j in range(0,10,1):
            for n in range(0,10,1):
            segment = accelerator.Segment([
                accelerator.Quadrupole(length=i, k1= , misalignment=(,) , name="Drift_AREASOLA1")
            ])
            assert [torch.tensor(
                    [
                        [1, i, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, i, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, i * igamma2, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                    ],
                    dtype=torch.float32, device="cpu") == segment.Drift_AREASOLA1.transfer_map(energy=j)]
"""
