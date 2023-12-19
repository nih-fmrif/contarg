import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

def plot_tail(tail, cdf):
    y = np.arange(len(tail)) / float(len(tail))
    plt.plot(tail, y)
    plt.plot(
        np.arange(tail.min(), tail.max(), 0.001),
        cdf(np.arange(tail.min(), tail.max(), 0.001)),
    )
    plt.show()


def _run_gpd_p(x, x0=0, side="upper", nx=260, fit_alpha=0.05, plot=False):
    """Fit tail with generalized pareto distribution to get p-value of x0.
    Based on Knijnenburg et al, 2009 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2687965/)
    Here we use a komogorov-smirnof test for equality of distributions
    instead of Anderson-Darling as used in their paper.
    Parameters
    ==========
    x: array
        Array of values containing the bootstrap or permutation distribution
    x0: float
        The value the distribution is being tested against
    side: string
        Specify the tail of the distribution to be tested must be one of ["upper", "lower"]
    nx: int
        Starting value for the number of excedences to begin counting down from
        while attempting to fit the GPD
    fit_alpha: float
        Alpha used to reject the null hypothesis that the tail of the data
        comes from the fitted GPD.
    Returns
    =======
    p: float
        fitted p-value
    """
    x = np.sort(x)
    fit_p = 0
    n = len(x)
    if nx > len(x):
        nx = len(x)
    if side == "upper":
        epc = np.count_nonzero(x >= x0)
    elif side == "lower":
        epc = np.count_nonzero(x <= x0)
    else:
        raise ValueError(f'side must be one of ["upper", "lower"], you provided {side}')
    if epc >= 10:
        # TODO: binomial estimate of this
        return (epc + 1) / (n + 1)
    while (fit_p < fit_alpha) & (nx > 10):
        nx -= 10
        if side == "upper":
            t = np.mean([x[-1 * nx], x[-1 * nx - 1]])
            tail = x[-1 * nx :] - t
        else:
            t = np.mean([x[nx], x[nx + 1]])
            tail = np.sort((x[:nx]) - t)
        fit_params = stats.genpareto.fit(tail)
        fitted_gpd = stats.genpareto(*fit_params)
        k = fitted_gpd.args[2]
        fit_stat, fit_p = stats.kstest(tail, fitted_gpd.cdf)
    if fit_p < fit_alpha:
        print(
            "Could not fit GPD to tail of distribution, returning empirical cdf based p.",
            flush=True,
        )
        return (epc + 1) / (n + 1)
        # raise Exception("Could not fit GPD to tail of distribution")

    if plot:
        plot_tail(tail, fitted_gpd.cdf)

    if side == "upper":
        p = nx / n * (1 - fitted_gpd.cdf(x0 - t))
        # If p == 0 and K > 0 then we're in a domain where
        # GPD is finite and unsuitable for extrapolation
        # In these cases, return the pvalue for the extreme of x,
        # which will be conservative
        if (p == 0) & (k > 0):
            p = nx / n * (1 - fitted_gpd.cdf(x.max() - t))
            if p == 0:
                return (epc + 1) / (n + 1)
                # raise Exception("p = 0")
        elif (p == 0) & (k <= 0):
            raise Exception("p=0 and k is not > 0")
    else:
        p = nx / n * (fitted_gpd.cdf(x0 - t))
        if (p == 0) & (k > 0):
            p = nx / n * (fitted_gpd.cdf(x.min() - t))
            if p == 0:
                return (epc + 1) / (n + 1)
                # raise Exception("p = 0")
        elif (p == 0) & (k <= 0):
            raise Exception("p=0 and k is not > 0")

    # return nx, t, fitted_gpd, p
    return p


def get_bs_p(a, x=0, side="double", axis=None):
    """Fit tail with generalized pareto distribution to get p-value of x0.
    Based on Knijnenburg et al, 2009 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2687965/)
    Here we use a komogorov-smirnof test for equality of distributions
    instead of Anderson-Darling as used in their paper. Optionally
    fit on an ndimensional array.
    Parameters
    ==========
    a: array
        Array of values containing the bootstrap or permutation distribution
    x0: float
        The value the distribution is being tested against
    side: string
        Specify the tail of the distribution to be tested must be one of ["upper", "lower"]
    axis: None or int
        Specifies the dimension along which the bootstraps/permutations are found
    Returns
    =======
    p: float or array
        Fitted p-value, returns a single value if axis is None
    """
    if axis is not None:
        new_shape = np.array(a.shape)[np.arange(len(a.shape)) != axis]
        a = a.reshape(a.shape[axis], -1).T
    else:
        a = a.reshape(1, -1)

    res = np.zeros(a.shape[0])
    for ii, aa in enumerate(a):
        if side == "double":
            res[ii] = (
                np.min((_run_gpd_p(aa, x, "upper"), _run_gpd_p(aa, x, "lower"))) * 2
            )
            if res[ii] > 1:
                res[ii] = 1
        elif side in ["upper", "lower"]:
            res[ii] = _run_gpd_p(aa, x, side)
        else:
            raise ValueError(
                f'side must be one of ["upper", "lower", "double"], you provided {side}'
            )
    if axis is None:
        return res[0]
    else:
        return res.T.reshape(new_shape)


def get_bs_stats(alpha, bs_series):
    qup = 1 - (alpha / 2)
    qdn = alpha / 2
    vrd = {}
    vrd[f"q_{qup:0.4f}"] = bs_series.quantile(qup)
    vrd[f"q_{qdn:0.4f}"] = bs_series.quantile(qdn)
    vrd["uncorrected_p"] = get_bs_p(bs_series.values, 0)
    vrd["sign"] = np.sign(bs_series.mean())
    vrd["mean_val"] = bs_series.mean()
    vrd["std_val"] = bs_series.std()
    return vrd


def get_bs_res(bs, alpha, side="double"):
    bs_res = {}
    bs_res["bs_mean"] = bs.mean(0).squeeze()
    bs_res["bs_std"] = bs.std(0).squeeze()
    bs_res["bs_sign"] = np.sign(bs_res["bs_mean"])
    ps = get_bs_p(bs, 0, side, axis=0).squeeze()
    bs_res["bs_uncorrected_p"] = ps
    bs_res["bs_p"] = multipletests(ps.flatten(), method="sidak")[1].reshape(ps.shape)
    bs_res["bs_sig"] = bs_res["bs_p"] < alpha

    bs_res["bs_signed_ps"] = bs_res["bs_p"] * bs_res["bs_sign"]

    return bs_res