import matplotlib; matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats, optimize

import numpy as np
import helpers
from save_data import Counts
import os


sns.set_style('whitegrid')
dirplots = os.path.join(os.curdir, 'plots')
dirdata = os.path.join(os.curdir, 'all_data')


def bpl(x, xc, a, b):
    """Broken power law"""
    n = (xc**(1-a)*(a-b) + b-1)/((a-1)*(b-1))
    y1 = (x**(-a)/n)[x < xc]
    y2 = (xc**(-a+b) * x**(-b)/n)[x >= xc]
    y = np.concatenate((y1, y2))
    return y


def bpl_sf(x, xc, a, b):
    """Broken power law survival function"""
    n = (xc**(1-a)*(a-b) + b-1)/((a-1)*(b-1))
    y1 = ((1-x**(1-a))/(n*(a-1)))[x < xc]
    y2 = ((1-xc**(1-a))/(n*(a-1)) +
          (xc**(1-a)-xc**(b-a)*x**(1-b))/(n*(b-1)))[x >= xc]
    y = np.concatenate((y1, y2))
    return 1-y


def fit_bpl(data):
    """Maximum likelihood fit to data. Doesn't always get what we expect"""
    counts = data.counts
    popt = optimize.fmin(lambda p: - np.sum(np.log(bpl(counts, *p))), (data.Ncat, 1., 2.))
    return popt


def plot_AD(data):
    """Anderson-Darling test, given a Counts object"""
    # Notice that the stats.anderson function does not admit lognorm
    # With KS, I checked that using empirical KS test is the same, so I use it here as well
    counts = data.counts
    Ncat = data.Ncat
    fiducial = data.generate_mc(100)

    ln_par = data.lognorm_par()
    ad_ln = helpers.my_ad([counts, np.random.lognormal(np.log(ln_par[2]), ln_par[0], size=Ncat)])
    # ad_ln = stats.anderson(np.log(counts), 'norm')
    print("AD lognorm: "+str(ad_ln))

    ad_data = [helpers.my_ad([np.log(counts), np.log(mc)]) for mc in fiducial]

    # Plot Anderson-Darling
    fig = plt.figure(figsize=[10, 6.18])
    plt.title('Anderson-Darling statistics')
    plt.hist(ad_data, bins=10, label='MC', alpha=0.5)
    # plt.axvline(ks_tree[0], c='Purple', label = 'Tree model')
    plt.axvline(ad_ln[0], c='Orange', label='Lognormal')
    plt.legend(loc='best')
    # plt.savefig(os.path.join('all_data', 'AD_'+name+'.png'))
    return


def plot_KS(data):
    """KS test, given a Counts object"""
    counts = data.counts
    fiducial = data.generate_mc(100)

    ks_ln = stats.kstest(counts, 'lognorm', args=data.lognorm_par())
    ks_data = [stats.ks_2samp(counts, mc)[0] for mc in fiducial]

    # Plot KS
    fig = plt.figure(figsize=[10, 6.18])
    plt.title('Kolmogorov-Smirnov statistics')
    plt.hist(ks_data, bins=10, label='MC', alpha=0.5)
    # plt.axvline(ks_tree[0], c='Purple', label = 'Tree model')
    plt.axvline(ks_ln[0], c='Orange', label = 'Lognormal')
    plt.legend(loc='best')
    # plt.savefig(os.path.join('all_data', 'KS_'+name+'.png'))
    return fig


def plot_KL(data):
    """Kullback-Leibler divergence, given a Dataset object.
    The 'true' distribution is the data one"""
    frequencies = data.frequencies
    Ncat = data.Ncat
    fiducial = data.generate_mc(100)

    sh, loc, sc = data.lognorm_par()
    freq_ln = [np.sort(stats.lognorm.rvs(sh, scale=sc, size=Ncat, random_state=s))[::-1]
               for s in range(1, 1001)]
    kl_ln = [stats.entropy(frequencies, r) for r in freq_ln]

    lengths = [min(Ncat, len(mc)) for mc in fiducial]  # Cut to the minimum Ncat
    kl_data = [stats.entropy(frequencies[:lengths[i]], mc[:lengths[i]]) for i, mc in enumerate(fiducial)]

    # Plot KL divergence. Use kdeplot instead of histogram
    fig = plt.figure(figsize=[10, 6.18])
    plt.title('Kullback-Leibler divergence')
    # plt.hist(kl_data, bins=10, normed=True, label='MC', alpha=0.5)
    # plt.hist(kl_ln, bins=10, normed=True, label='Lognormal', alpha=0.5, color='Blue')
    sns.kdeplot(np.array(kl_data), label='MC', alpha=0.6, color='Blue')
    sns.kdeplot(np.array(kl_ln), label='Lognormal', alpha=0.6, color='Orange')
    plt.xlim(xmin=0.)
    # plt.axvline(ks_tree[0], c='Purple', label = 'Tree model')
    # plt.axvline(kl_ln, c='Orange', label = 'Lognormal')
    plt.legend(loc='best')
    # plt.savefig(os.path.join('all_data', 'KL_'+data.name+'.png'))
    return


def plot_loglog(data):
    """Plot data, given a Counts object"""
    counts = data.counts
    Ncat = data.Ncat
    ranks = data.ranks
    fiducial = data.generate_mc(100)
    cdf = data.get_cdf()

    fig = plt.figure(figsize=[10, 6.18])
    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([.5, max(counts)*10.])
    ax.set_ylim([1., Ncat*1.2])

    for f in fiducial:
        ax.plot(f, np.arange(1, len(f)+1),
                'o', ms=3, color='Gray', alpha=0.1, rasterized=True)

    ax.plot(counts, ranks,
            's', ms=3, color='Orange', rasterized=True,
            label='Data')

    ax.plot(cdf[0], Ncat*cdf[1],
            ls='--', color='Magenta', linewidth=3, label='Tree distr.')

    ax.plot(counts, Ncat*data.lognorm_sf(),
            ls='-.', color='Blue', linewidth=3, label='Lognormal')

    par = fit_bpl(data)
    ax.plot(np.sort(counts), Ncat*bpl_sf(np.sort(counts), *par),
            ls=':', color='Crimson', linewidth=3, label='Broken PL')

    avg = helpers.avg_mc(fiducial, counts)
    Nc = len(avg)
    ax.plot(avg, np.arange(1, Nc+1),
            ls='-', color='Lime', linewidth=3, label='Average')

    gray_patch = mpatches.Patch(color='Gray', label='MC')
    orange_patch = mpatches.Patch(color='Orange', label='Data')
    green_patch = mpatches.Patch(color='Lime', label='Average')
    magenta_patch = mpatches.Patch(color='Magenta', label='Tree distr.')
    blue_patch = mpatches.Patch(color='Blue', label='Lognormal')
    brown_patch = mpatches.Patch(color='Crimson', label='Broken PL')
    plt.legend(handles=[gray_patch, orange_patch, green_patch,
                        magenta_patch, blue_patch, brown_patch])
    return fig


def plot_loglin(data):
    """Plot data, given a Counts object"""
    counts = data.counts
    Ncat = data.Ncat
    ranks = data.ranks
    fiducial = data.generate_mc(100)
    cdf = data.get_cdf()

    fig = plt.figure(figsize=[10, 6.18])
    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_xlim([.5, max(counts)*10.])
    ax.set_ylim([1., Ncat*1.2])

    for f in fiducial:
        ax.plot(f, np.arange(1, len(f)+1),
                'o', ms=3, color='Gray', alpha=0.1, rasterized=True)

    ax.plot(counts, ranks,
            's', ms=3, color='Orange', rasterized=True,
            label='Data')

    ax.plot(cdf[0], Ncat*cdf[1],
            ls='--', color='Magenta', linewidth=3, label='Tree distr.')

    ax.plot(counts, Ncat*data.lognorm_sf(),
            ls='-.', color='Blue', linewidth=3, label='Lognormal')

    par = fit_bpl(data)
    ax.plot(np.sort(counts), Ncat*bpl_sf(np.sort(counts), *par),
            ls=':', color='Crimson', linewidth=3, label='Broken PL')

    avg = helpers.avg_mc(fiducial, counts)
    Nc = len(avg)
    ax.plot(avg, np.arange(1, Nc+1),
            ls='-', color='Lime', linewidth=3, label='Average')

    gray_patch = mpatches.Patch(color='Gray', label='MC')
    orange_patch = mpatches.Patch(color='Orange', label='Data')
    green_patch = mpatches.Patch(color='Lime', label='Average')
    magenta_patch = mpatches.Patch(color='Magenta', label='Tree distr.')
    blue_patch = mpatches.Patch(color='Blue', label='Lognormal')
    brown_patch = mpatches.Patch(color='Crimson', label='Broken PL')
    plt.legend(handles=[gray_patch, orange_patch, green_patch,
                        magenta_patch, blue_patch, brown_patch])
    return fig


def main():
    names = ('sample',)
    sigma_Ncat = np.empty((len(names), 4))

    for i, name in enumerate(names):
        print('\nDoing '+name+'.....')
        data = Counts(os.path.join(os.curdir, 'counts_'+name+'.dat'))

        counts = data.counts
        ranks = data.ranks
        Ncat = data.Ncat
        Nitems = data.Nitems

        factor = data.get_factor()
        print("Ncat = ", Ncat, "\tNitems = ", Nitems)
        print("factor = ", factor)

        # sigma_Ncat[i, 0] = Ncat
        # sigma_Ncat[i, 1] = data.lognorm_par()[0]
        # sigma_Ncat[i, 2] = np.var(np.log(counts))
        # # Check that we have all non-zero categories
        # print("Categories with zero count:",
        #       np.sum([m[m == 0].shape for m in data.fiducial]))
        # sigma_Ncat[i, 3] = np.mean([np.var(np.log(m)) for m in data.fiducial])

        chao_est, chao_sigma = helpers.chao(data)
        print('Chao estimator: ', chao_est)
        print('Chao std: ', chao_sigma)
        print('Our estimated Q: ', factor * data.Ncat)
        print('Ratio Q/Chao: ', factor * data.Ncat/chao_est)
        print('Sigma/Est for Chao: ', chao_sigma/chao_est)
        fig = plot_KL(data)
        plt.savefig('KL_plot.png')
        fig = plot_loglog(data)
        plt.savefig('loglog_plot.png')
        fig = plot_loglin(data)
        plt.savefig('loglin_plot.png')

if __name__ == "__main__":
    main()
