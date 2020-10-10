#!/usr/bin/env python3

import warnings
import matplotlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import nptyping as npt

from matplotlib import axes
from typing import Tuple, Any
from scipy.stats._continuous_distns import _distn_names as _cts_dist_names
from scipy.stats._discrete_distns import _distn_names as _disc_dist_names

# There appear to be issues with the levy stable dist fit method
_cts_dist_names.remove('levy_stable')
matplotlib.style.use('ggplot')
sns.set(palette='muted')




class Error(Exception):
    pass




class ConvexCombinationError(Error):
    """
    Error raised when input is not a convex combination, i.e. sum != 1.
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message



class mixture_rv(stats.rv_continuous):
    """
    Create a mixture random variable from other random variables
    Convex combinations of pdfs is also a pdf

    Note:
        Convex combination of the pdfs in submodels,
        convex_comb is such that sum(convex_comb)=1.
    """
    def __init__(self, submodels, convex_comb, *args, **kwds):
        super().__init__(*args, **kwds)
        self.submodels = submodels
        if sum(convex_comb) != 1:
            raise ConvexCombinationError(f"coeffs = {convex_comb}", f"Sum of coefficients is {sum(convex_comb)}. Sum must be 1.")
        self.coeffs = convex_comb

    def _pdf(self, x):
        return sum([coeff*model.pdf(x) for model,coeff in zip(self.submodels, self.coeffs)])




class exp_rv(stats.rv_continuous):
    """
    Create your own random variable object extending
        the rv_continuous class

    To obtain an object of this class:

        exponential_rv = exp_rv(name='exponential', a=0., b=float('inf'))
            
        where [a,b] is the domain and name is the desired name for this object
    
    Example creation and run of cts_dist_fit_plot:
        exponential_rv = exp_rv(name='exponential', a=0., b=float('inf'))
        test_samples = exponential_rv(k=0.5).rvs(size=2000)
        cts_dist_fit_plot(data=test_samples)
    """
    def _pdf(self, x, k):
        return k*np.exp(-k*x)




class benktander_type2(stats.rv_continuous):
    """
    Benktander Type II Distribution

    Description: The Benktander type II distribution, also called the Benktander distribution of the second kind, is one of two distributions introduced by Gunnar Benktander (1970) to model heavy-tailed losses commonly found in non-life/casualty actuarial science, using various forms of mean excess functions (Benktander & Segerdahl 1960). This distribution is "close" to the Weibull distribution (Kleiber & Kotz 2003).

    Parameters:
        k > 0, 
        0 < m <= 1

    Support:
        x >= 1

    Example:
        benktander_type2_rv = benktander_type2(name='benktander_type2', a=1., b=float('inf'))
    """
    def _pdf(self, x, k, m):
        return (k*(x**m)-m+1)*(x**(m-2))*np.exp((m/k)*(1-(x**m)))




def plot_cts_distribution(
    distribution: stats.rv_continuous, 
    epsilon: float = 5e-5,
    epsilon_end: float = 1e-10,
    from_samples: bool = False,
    num_samples: int = 10000
) -> None:
    """
    Plot the continuous distribution using seaborn and matplotlib
    See scipy.stats for cts distributions: 
        https://docs.scipy.org/doc/scipy/reference/stats.html 

    epsilon              ->  Dist between plot points
    epsilon_end          ->  Plot on interval [x,y] where P(X<x) = epsilon_end to P(X<y) = 1 - epsilon_end
    from_samples = True  ->  Plot from random sampling
                 = False ->  Plot the PDF
    num_samples          ->  If samples true gives number of samples

    Example:

        plot_cts_distribution(stats.arcsine(), epsilon_end=1e-2)
        plot_cts_distribution(stats.arcsine(), samples=True)

        User-defined:

            plot_cts_distribution(exponential_rv(k=0.5))
            plot_cts_distribution(exponential_rv(k=0.5),from_samples=True,num_samples=1000)
    """
    if from_samples:
        rv_samples = distribution.rvs(size=num_samples)
        ax = sns.distplot(rv_samples, color="m")
        plt.title('Samples')
    else: 
        x = np.linspace(distribution.ppf(0+epsilon_end), distribution.ppf(1-epsilon_end), int(1/epsilon))
        df = pd.DataFrame({'Values':x,'Probability':distribution.pdf(x)})
        ax = sns.lineplot(x='Values',y='Probability',data=df)
        plt.title('Density')
    plt.show()
    return




def plot_dis_distribution(
    distribution: stats.rv_discrete,
    epsilon_end: float = 1e-10,
    from_samples: bool = False,
    num_samples: int = 10000
) -> None:
    """
    Plot the discrete distribution using seaborn and matplotlib
    See scipy.stats for discrete distributions:
        https://docs.scipy.org/doc/scipy/reference/stats.html

    epsilon_end     ->  Plot on interval [x,y] where P(X<x) = epsilon_end to P(X<y) = 1 - epsilon_end
    """
    if from_samples:
        pass
    else:
        x = np.arange(distribution.ppf(0+epsilon_end),distribution.ppf(1-epsilon_end))
        fig, ax = plt.subplots(1,1)
        ax.plot(x, distribution.pmf(x), 'bo', ms=8, label='discrete-pmf')
        plt.show()
    return




def cts_dist_fit(
    data: npt.NDArray[Any,Any],
    bins: int = 200,
    ax: axes = None
) -> Tuple:
    """
    Test all continuous distributions in scipy.stats._continuous_distns

    data    -> numpy array data object of data to fit
    bins    -> binning of data
    ax      -> matplotlib.axes for plotting all fitted distns

    Example without Plotting:

        test_samples = stats.arcsine(loc=2.0,scale=2.0).rvs(size=2000)
        best_dist, best_params, best_sse = cts_dist_fit(test_samples)
    """
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    x = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
    best_dist, best_params, best_sse = None, None, float('inf')
    i = 1
    num_dists = len(_cts_dist_names)
    for distribution in sorted(_cts_dist_names):
        print(f'Fitting {distribution} distribution [({i}) of ({num_dists})]...')
        print(f'Best so far is {best_dist} at {best_sse}...')
        stats_dist_object = getattr(stats,distribution)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                *args,loc,scale = stats_dist_object.fit(data)
                pdf = stats_dist_object.pdf(x, loc=loc, scale=scale, *args)
                sse = np.sum(np.power(hist - pdf, 2.0))
                print(f'Fitted {distribution} with sse = {sse}\n')
                try:
                    if ax:
                        df = pd.Series(pdf,x)
                        sns.lineplot(data=df,ax=ax)
                except Exception:
                    pass
                if 0 < sse < best_sse:
                    best_dist = distribution
                    best_params = (*args,loc,scale)
                    best_sse = sse
        except Exception:
            print(f'Could not fit {distribution} distribution.')
        i += 1
    return (best_dist,best_params,best_sse)




def cts_dist_fit_plot(
    data: npt.NDArray[Any,Any],
    bins: int = 200,
    ax: axes = None
) -> Tuple:
    """
    Display all fitted distributions to the sample data.

    data    ->  data to be fitted
    bins    ->  num bins for histogram
    ax      ->  axes to display plots on

    Example:

        test_samples = stats.arcsine(loc=2.0,scale=2.0).rvs(size=2000)
        cts_dist_fit_plot(data=test_samples)
    """
    dataframe = pd.Series(data=data)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    sns.distplot(dataframe, bins=bins, norm_hist=True, kde=False, ax=axes[0])
    dataYLim = axes[0].get_ylim()
    best_dist, best_params, best_sse = cts_dist_fit(data, bins=bins, ax=axes[0])
    axes[0].set_ylim(dataYLim)
    *args,loc,scale = best_params
    sns.distplot(data, bins=bins, norm_hist=True, kde=False, ax=axes[1])
    best_dist_object_unfrozen = getattr(stats,best_dist)
    best_dist_object = best_dist_object_unfrozen(*args,loc=loc,scale=scale)
    x = np.linspace(best_dist_object.ppf(0.01), best_dist_object.ppf(0.99), 10000)
    df = pd.Series(best_dist_object.pdf(x),x)
    sns.lineplot(data=df, ax=axes[1])
    param_names = (best_dist_object_unfrozen.shapes + ', loc, scale').split(', ') if best_dist_object_unfrozen.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_params)])
    dist_str = '{}({})'.format(best_dist, param_str)
    axes[0].set_title(u'Best Fit Distribution for Sample Data \n' + dist_str)
    plt.show()
    return (best_dist, best_params, best_sse)




if __name__ == '__main__':
    # benktander_type2_rv = benktander_type2(name='benktander_type2', a=1., b=float('inf'))
    # test_samples = benktander_type2_rv(4,0.5).rvs(size=2000)
    # cts_dist_fit_plot(data=test_samples)
    submodels = [stats.norm(loc=-1,scale=0.2),stats.norm(loc=1,scale=0.2),stats.norm(loc=0,scale=0.6)]
    coeffs = [0.2,0.2,0.7]
    mixture_normal_rv = mixture_rv(submodels,coeffs,name='normal_mixture')
    plot_cts_distribution(mixture_normal_rv)