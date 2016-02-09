from scipy import optimize, stats
import numpy as np
import helpers


# Run once to compile the functions
print("Compiling the functions...")
print(helpers.one_trial_div2(20, 60))
print(helpers.get_tree_fudge_multi(20, 60, 2))
print(helpers.Ptree(20, 10))
print(helpers.f_empty(20, 10))
print(helpers.n_filled(20, 10))
print(helpers.cdf(20))
print(helpers.get_mc(20, 60, 2, 10))


class Counts(object):
    """Class that reads a counts filename and holds
    all the operations we want to do"""

    def __init__(self, fname):
        """Read a file of (sorted) counts"""
        self.counts = np.sort(np.loadtxt(fname))[::-1]
        self.Ncat = len(self.counts)
        self.Nitems = np.sum(self.counts)
        self.frequencies = self.counts/self.Nitems
        self.ranks = np.arange(1, self.Ncat+1)

    def get_cdf(self):
        cdf_counts = self.Nitems * 0.5**np.arange(1, self.Ncat+1)
        cdf = helpers.cdf(self.Ncat)
        return np.vstack((cdf_counts, cdf))

    def get_factor(self):
        """Gives the expected Q/K. It should always converge."""
        res = optimize.fsolve(lambda x: helpers.n_filled(int(x*self.Ncat), self.Nitems) - self.Ncat,
                              1.1, xtol=0.001, full_output=True)
        return res[0][0]

    def generate_mc(self, n_trials, tol=1.e-2):
        factor = self.get_factor()
        mc_fudge = helpers.get_mc(self.Ncat, self.Nitems, n_trials, factor, tol)
        return mc_fudge

    def lognorm_sf(self):
        """Gives the survival function of the best lognormal fit"""
        shape, loc, scale = stats.lognorm.fit(self.counts, floc=0)
        return stats.lognorm.sf(self.counts, shape, loc=0, scale=scale)

    def lognorm_par(self):
        """Gives MLE parameters of a lognormal fit"""
        shape, loc, scale = stats.lognorm.fit(self.counts, floc=0)
        return shape, loc, scale
