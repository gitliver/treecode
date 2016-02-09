import numpy as np
from scipy import stats
import numba


@numba.jit(nopython=False)
def one_trial_div2(Ncat, Nitems, seed=None):
    counts = np.empty(Ncat)
    counts[0] = Nitems
    prng = np.random.RandomState(seed)
    for i in range(Ncat - 1):
        idx = prng.randint(i + 1)
        counts[i + 1] = 0.5 * counts[idx]
        counts[idx] *= 0.5
    return np.sort(counts)[::-1]


@numba.jit(nopython=False)
def get_fiducial_mc(Ncat, Nitems, Ntrials, seeds):
    """Gives Ntrials Montecarlo realizations, given Ncat, Nitems.
    Seeds is a vector of Ntrials seeds for the random number generator."""
    res = np.empty((Ntrials, Ncat), dtype=np.float64)
    for i in range(Ntrials):
        prng = np.random.RandomState(seeds[i])
        counts = one_trial_div2(Ncat, Nitems, seeds[i])
        uni_draw = prng.uniform(-1., 1., Ncat)
        new_counts = counts * 2**(uni_draw)
        prob_vec = new_counts/np.sum(new_counts)
        mult_draw = prng.multinomial(Nitems, prob_vec)
        # res.append(np.sort(mult_draw)[::-1])
        res[i] = np.sort(mult_draw)[::-1]
    return res


@numba.jit(nopython=False)
def get_tree_fudge_multi(Ncat, Nitems, Ntrials):
    res = np.empty((Ntrials, Ncat))
    prng = np.random.RandomState()
    for i in range(Ntrials):
        counts = one_trial_div2(Ncat, Nitems)
        uni_draw = prng.uniform(-1., 1., Ncat)
        new_counts = counts * 2**(uni_draw)
        prob_vec = new_counts/np.sum(new_counts)
        mult_draw = prng.multinomial(Nitems, prob_vec)
        res[i] = np.sort(mult_draw)[::-1]
    return res


@numba.jit
def scycle(N, k):
    """Computes absolute value of Stirling number of first kind"""
    if (k < 0) or (k > N):
        return 0
    elif k == 0 or k == N:
        return 1
    else:
        a = np.zeros(k+1, dtype=np.uint64)
        a[1] = 1
        for i in range(1, N):
            w = a[1]
            a[1] = i * w
            for j in range(2, min(i, k)+1):
                v = a[j]
                a[j] = w + i * v
                w = v
            if i < k:
                a[i+1] = 1
        s = a[k]
    return s


@numba.jit(nopython=True)
def Ptree(N, k):
    """Computes the probability that a random leaf,
    in a tree with N leaves, has distance k from the root.
    Numba gives a speedup of a factor 50.
    Using nopython=True gives me another factor of 30!"""
    if (k < 0) or (k >= N):
        return 0
    elif k == 0 and k == N:
        return 1
    else:
        a = np.zeros(k+1, dtype=np.float64)
        a[1] = 1
        for i in range(2, N):
            w = a[1]
            a[1] = w * (i-1)/(i+1)
            for j in range(2, min(i, k)+1):
                v = a[j]
                a[j] = (2*w + (i-1) * v)/(i+1)
                w = v
        s = a[k]
    return s


@numba.jit
def cdf(Ncat):
    """Calculates 1-CDF of the tree model.
    The output is a np.array of frequencies."""
    max_sum = int(5 * np.log2(Ncat))
    sf = np.ones(Ncat, dtype=np.float64)
    sf[0] = Ptree(Ncat, 1)
    for q in range(1, max_sum+1):
        sf[q] = sf[q-1] + Ptree(Ncat, q+1)
    return sf


@numba.jit
def chao(data):
    """Implements the Chao estimator, to predict the number
    of categories in the population from a sample.
    It is generally an underestimate of the true number.
    The input is a numpy array with set of observations (counts per category).
    Outputs the mean of the estimator and its standard deviation"""
    counts = data.counts
    Ncat = data.Ncat  # Number of categories (they are all non-empty)
    Nitems = data.Nitems
    n1 = np.sum(counts == 1)
    n2 = np.sum(counts == 2)
    # n3 = np.sum(counts == 3) 
    # m1 = 2*n2/n1
    # m2 = 6*n3/n1
    # check1 = (m2 > m1**2)
    # check2 = (Nitems*m1 > m2)
    # if not check1:
    #     print('We have m2 < m1^2')
    # elif not check2:
    #     print('We have N m1 < m2')
    Ncat_est = Ncat + 0.5 * n1*(n1-1)/(n2+1)
    var_Ncat = (0.5 * n1*(n1-1)/(n2+1)
        + 0.25 * n1 * (2*n1-1)**2/(n2+1)**2
        + 0.25 * n1**2 * n2 * (n1-1)**2/(n2+1)**4)
    # var_Ncat = (0.25*(n1**2 * (n1-1)**2 * n2)/(n2+1)**4
    #             + 0.25 * (n1 * (2*n1-1)**2)/(n2+1)**2
    #             + 0.5 * (n1 * (n1-1))/(n2+1))
    return (Ncat_est, np.sqrt(var_Ncat))


@numba.jit(nopython=True)
def f_empty(Ncat, Nitems):
    """Fraction of empty categories given Ncat, Nitems.
    A bit slow for large Ncat."""
    s = 0.
    max_sum = int(2 * np.log2(Ncat*Nitems))
    for q in range(1, max_sum+1):
        s += Ptree(Ncat, q) * (1 - 2.**(-q))**Nitems
    return s


@numba.jit(nopython=True)
def n_filled(Ncat, Nitems):
    """Average number of filled categories given Ncat, Nitems.
    A bit slow for large Ncat."""
    s = 0.
    max_sum = int(2 * np.log2(Ncat*Nitems))
    for q in range(1, max_sum+1):
        s += Ptree(Ncat, q) * (1 - 2.**(-q))**Nitems
    return Ncat * (1.-s)


#@numba.jit()
def avg_mc(fiducials, counts):
    """Calculates the average over the fiducial realizations.
    They might differ in the number of categories,
    so I will consider the minimum."""
    Ncat = min((len(counts), min([len(f) for f in fiducials])))
    fid_T = np.array([f[:Ncat] for f in fiducials]).transpose()  # shape(Ncat, Ntrials)
    avg = []
    for i in range(Ncat):
        avg.append(np.mean(fid_T[i]))
    return np.array(avg)


@numba.jit('pyobject[:](uint64, uint64, uint16, f4, f4)', nopython=False)
def get_mc(Ncat, Nitems, Nmc, factor, tol=1.e-2):
    """Gives Nmc Montecarlo realizations, given Ncat, Nitems.
    We need to multiply by factor to fill, on average,
    Ncat by multinomial sampling.
    tol is the relative uncertainty we allow on the number of categories."""
    mc = np.empty(Nmc, dtype=object)
    num = 0
    seed = 1
    while num < Nmc:
        mc_fudge = get_fiducial_mc(int(factor * Ncat), Nitems, 1, [seed])
        # Check if we fill the correct number of categories
        pos_idx = mc_fudge[0] > 0
        frac = pos_idx.sum()/Ncat
        # Get the ones which have a number of filled categories within 1% of the data
        if np.isclose(frac, 1., rtol=1e-2):
            mc[num] = mc_fudge[0][pos_idx]
            num += 1
        seed += 1
    return mc


def my_ad(samples, midrank=True):
    """Anderson-Darling k-sample test.
    It is from scipy.stats, I had to remove the p-value calculation as it gives overflow."""
    k = len(samples)
    if (k < 2):
        raise ValueError("anderson_ksamp needs at least two samples")

    samples = list(map(np.asarray, samples))
    Z = np.sort(np.hstack(samples))
    N = Z.size
    Zstar = np.unique(Z)
    if Zstar.size < 2:
        raise ValueError("anderson_ksamp needs more than one distinct "
                         "observation")

    n = np.array([sample.size for sample in samples])
    if any(n == 0):
        raise ValueError("anderson_ksamp encountered sample without "
                         "observations")

    if midrank:
        A2kN = stats.morestats._anderson_ksamp_midrank(samples, Z, Zstar, k, n, N)
    else:
        A2kN = stats.morestats._anderson_ksamp_right(samples, Z, Zstar, k, n, N)

    H = (1. / n).sum()
    hs_cs = (1. / np.arange(N - 1, 1, -1)).cumsum()
    h = hs_cs[-1] + 1
    g = (hs_cs / np.arange(2, N)).sum()

    a = (4*g - 6) * (k - 1) + (10 - 6*g)*H
    b = (2*g - 4)*k**2 + 8*h*k + (2*g - 14*h - 4)*H - 8*h + 4*g - 6
    c = (6*h + 2*g - 2)*k**2 + (4*h - 4*g + 6)*k + (2*h - 6)*H + 4*h
    d = (2*h + 6)*k**2 - 4*h*k
    sigmasq = (a*N**3 + b*N**2 + c*N + d) / ((N - 1.) * (N - 2.) * (N - 3.))
    m = k - 1
    A2 = (A2kN - m) / np.sqrt(sigmasq)
    return A2
