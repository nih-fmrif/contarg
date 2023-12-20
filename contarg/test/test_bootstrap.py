import numpy as np
from contarg.bootstrap import _run_gpd_p

def test_run_gpd_p():
    np.random.seed(seed=1)
    assert _run_gpd_p(np.random.standard_normal(1000) + 3, side="lower") < 0.05
    assert _run_gpd_p(np.random.standard_normal(1000) + 3, side="upper") > 0.95
    assert _run_gpd_p(np.random.standard_normal(1000) - 3, side="lower") > 0.95
    assert _run_gpd_p(np.random.standard_normal(1000) - 3, side="upper") < 0.05


if __name__ == "__main__":
    test_run_gpd_p()