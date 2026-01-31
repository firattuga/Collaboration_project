import numpy as np
import pytest

@pytest.fixture
def dummy_tracks():
    return {
        0: {
            "path": [
                (0, 0.1, 0.2, 1.0, 1.0),
                (1, 0.15, 0.25, 1.1, 2.0),
                (2, 0.2, 0.3, 1.2, 3.0),
            ],
            "fit": {"xc": 0.0, "yc": 0.0, "R": 1.0},
        },
        1: {
            "path": [
                (0, -0.1, 0.1, 1.0, 1.5),
                (1, -0.15, 0.15, 1.1, 2.5),
            ],
            "fit": {"xc": 0.1, "yc": -0.1, "R": 1.2},
        },
    }


@pytest.fixture
def dummy_trajectories():
    return {
        0: np.array([[0, 0, 0], [0.1, 0.1, 1.0], [0.2, 0.2, 1.2]]),
        1: np.array([[0, 0, 0], [-0.1, 0.1, 1.0], [-0.2, 0.2, 1.2]]),
    }


@pytest.fixture
def dummy_hits_3d():
    return {
        0: [(0.1, 0.1, 1.0), (0.2, 0.2, 1.2)],
        1: [(-0.1, 0.1, 1.0)],
    }
