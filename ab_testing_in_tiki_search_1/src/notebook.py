#!/usr/bin/python

import os
import sys
import math
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

def generate(N=20000, experiments=2000, mu=5, sigma2=1.3, \
        success_rate=0.02, uplift=0.03, beta=100):
    # generate {experiments} number of views distributions each containing {N} users
    views_a_1 = np.absolute(np.exp(stats.norm(mu, sigma2).rvs(experiments * N)) \
        .astype(np.int64).reshape(experiments, N) + 1)
    views_a_2 = np.absolute(np.exp(stats.norm(mu, sigma2).rvs(experiments * N)) \
        .astype(np.int64).reshape(experiments, N) + 1)
    views_b = np.absolute(np.exp(stats.norm(mu, sigma2).rvs(experiments * N)) \
        .astype(np.int64).reshape(experiments, N) + 1)

    # ground truth CTR for groups A1 and A2
    alpha_a = success_rate * beta / (1 - success_rate)
    success_rate_a_1 = stats.beta(alpha_a, beta).rvs(experiments * N) \
        .reshape(experiments, N)
    success_rate_a_2 = stats.beta(alpha_a, beta).rvs(experiments * N) \
        .reshape(experiments, N)

    # ground truth CTR for group B (with uplift)
    alpha_b = success_rate * (1 + uplift) * beta / (1 - success_rate * (1 + uplift))
    success_rate_b = stats.beta(alpha_b, beta).rvs(experiments * N) \
        .reshape(experiments, N)

    # clicks distribution
    clicks_a_1 = stats.binom(n=views_a_1, p=success_rate_a_1).rvs()
    clicks_a_2 = stats.binom(n=views_a_2, p=success_rate_a_2).rvs()
    clicks_b = stats.binom(n=views_b, p=success_rate_b).rvs()

    return views_a_1, views_a_2, views_b, \
        success_rate_a_1, success_rate_a_2, success_rate_b, \
        clicks_a_1, clicks_a_2, clicks_b

def t_test(a, b):
    result = list(map(lambda x: stats.ttest_ind(x[0], x[1]).pvalue, zip(a, b)))
    return np.array(result)

def mannwhitney(a, b):
    result = list(map(lambda x: stats.mannwhitneyu(x[0], x[1], \
        alternative='two-sided').pvalue, zip(a, b)))
    return np.array(result)

colors = sns.color_palette("deep")

def plot_cdf(data: np.ndarray, label: str, ax: Axes, color: str = colors[0], linewidth=3):
    sorted_data = np.sort(data)
    position = stats.rankdata(sorted_data, method='ordinal')
    cdf = position / data.shape[0]
    sorted_data = np.hstack((sorted_data, 1))
    cdf = np.hstack((cdf, 1))
    return ax.plot(sorted_data, cdf, color=color, linestyle='solid', \
        label=label, linewidth=linewidth)

### Example ###
# views_a_1, views_a_2, views_b, \
    # success_rate_a_1, success_rate_a_2, success_rate_b, \
    # clicks_a_1, clicks_a_2, clicks_b = generate()

# ctrs_a_1 = clicks_a_1.astype(np.float) / views_a_1
# ctrs_a_2 = clicks_a_2.astype(np.float) / views_a_2
# ctrs_b = clicks_b.astype(np.float) / views_b

# p_values_t_test = t_test(ctrs_a_1, ctrs_a_2)
# p_values_mannwhitney = mannwhitney(ctrs_a_1, ctrs_a_2)

# def plot_pvalues(p_values, title):
    # fig, axes = plt.subplots(ncols=1, nrows=1)
    # p_values_distr = sns.histplot(p_values, bins=np.linspace(0, 1, 20), \
        # kde=False, ax=axes, stat='probability')
    # axes.set_xlim((0, 1))
    # axes.set_title(title)
    # axes.set(xlabel = 'p-value')
    # plt.tight_layout()
    # plt.show()

# plot_pvalues(p_values_t_test, 'p-values in A/A test (T-Test)')
# plot_pvalues(p_values_mannwhitney, 'p-values in A/A test (Mann-Whitney)')

# def plot_pvalues_cdf(p_values, title):
    # fig, ax = plt.subplots()
    # gr = ax.grid(True)
    # xlim = ax.set_xlim(-0.05,1.02)
    # ylim = ax.set_ylim(-0.02,1.02)
    # ax.axvline(0.05, color='k', alpha=0.5)
    # ax.set_xlabel(r'$\alpha$')
    # ax.set_ylabel('Sensitivity')
    # ax.set_title(title)
    # cdf = plot_cdf(p_values, '', ax)
    # plt.show()

# plot_pvalues_cdf(p_values_t_test, 'p-values CDF in A/A test (T-Test)')
# plot_pvalues_cdf(p_values_mannwhitney, 'p-values CDF in A/A test (Mann-Whitney)')

# p_values_t_test = t_test(ctrs_a_1, ctrs_b)
# p_values_mannwhitney = mannwhitney(ctrs_a_1, ctrs_b)

# plot_pvalues(p_values_t_test, 'p-values in A/B test (T-Test)')
# plot_pvalues(p_values_mannwhitney, 'p-values in A/B test (Mann-Whitney)')

# plot_pvalues_cdf(p_values_t_test, 'p-values CDF in A/B test (T-Test)')
# plot_pvalues_cdf(p_values_mannwhitney, 'p-values CDF in A/B test (Mann-Whitney)')

# sys.exit(0)
### END OF Example ###

def plot_summary(data, ground_truth_ctr, views, sigma2, beta, mu, N):
    cdf_h1_title = 'p-value if H1 is correct (Sensitivity)'
    cdf_h0_title = 'p-value if H0 is correct (FPR)'
    
    # create layout
    fig = plt.figure(constrained_layout=False, figsize=(3 * 3.5, 3.5 * 3), dpi = 100)
    gs = fig.add_gridspec(4, 3)
    
    # fill the layout
    ax_h1 = fig.add_subplot(gs[:2, :2]) # sensitivity in A/B
    ax_h0 = fig.add_subplot(gs[0, 2]) # FPR in A/A
    ax_views = fig.add_subplot(gs[1, 2]) # views in A/A
    ax_clicks = fig.add_subplot(gs[2, 2]) # ctr in A/A
    ax_powers = fig.add_subplot(gs[2, :2]) # sensitivity at alpha = 0.05
    ax_fpr = fig.add_subplot(gs[3, :2]) # FPR at alpha = 0.05
    ax_hint = fig.add_subplot(gs[3, 2]) # current parameters hint
    
    # adjustments
    fig.subplots_adjust(left=0.2, wspace=0.3, hspace=0.4)
    
    # diagonals at h1 and h0 for simpler comparison
    ax_h1.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), 'k', alpha=0.1)
    ax_h0.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), 'k', alpha=0.1)

    ax_h1.set_title(cdf_h1_title)
    ax_h0.set_title(cdf_h0_title)

    ax_h1.axvline(0.05, color='k', alpha=0.5)

    # plot cdf for all
    for title, (ab_pvals, aa_pvals, color) in data.items():
        plot_cdf(ab_pvals, title, ax_h1, color, linewidth=3)
        plot_cdf(aa_pvals, title, ax_h0, color, linewidth=1.5)
    
    ax_h1.grid(True)
    ax_h0.grid(True)
    
    ax_powers.set_title('Sensitivity')
    ax_fpr.set_title(r'False Positive rate at $\alpha = 0.05$')
    tests_powers = []
    tests_fprs = []
    tests_labels = []
    tests_colours = []
    
    for title, (ab_pvals,aa_pvals, color) in data.items():
        tests_labels.append(title)
        tests_colours.append(color)
        tests_powers.append(np.mean(ab_pvals < 0.05))
        tests_fprs.append(np.mean(aa_pvals < 0.05))
    ax_powers.barh(np.array(tests_labels), np.array(tests_powers), color=np.array(tests_colours))        
    ax_fpr.barh(np.array(tests_labels), np.array(tests_fprs), color=np.array(tests_colours))

    views_p99 = np.percentile(views[:100].ravel(), 99)
    sns.histplot(views.ravel(),
                 bins=np.linspace(0, views_p99, 100),
                 ax=ax_views,
                 kde=False,
                 stat='probability').set(ylabel=None)
    ax_views.set_title(f'Views, P99 = {views_p99:7.1f}')

    success_rate_p99 = np.percentile(ground_truth_ctr[:100].ravel(), 99)
    sns.histplot(ground_truth_ctr[:10].ravel(),
                 bins=np.linspace(0, success_rate_p99, 100),
                 ax=ax_clicks,
                 kde=False,
                 stat='probability').set(ylabel=None)
    success_rate_std = ground_truth_ctr[:100].flatten().std()
    ax_clicks.set_title(f'Ground truth CTR, std = {success_rate_std:2.3f}')

    ax_hint.axis('off')
    ax_hint.text(0.5, 0.5, f'sigma2 = {sigma2}\nbeta = {beta}\nmu = {mu}\nN = {N}', \
        ha='center', va='center', size='xx-large')

    plt.close()
    return fig

### GIF processing ###

from PIL import Image
import imageio
import re

def to_gif(png_list, gif_name, delete_after=True):
    frames = []
    for i in png_list:
        new_frame = Image.open(i)
        frames.append(new_frame)
    frames[0].save(gif_name, format='GIF', append_images=frames[1:], save_all=True, duration=800, loop=0)
    if delete_after:
        for i in png_list:
            os.remove(i)

### END GIF processing ###

sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 4.5]
beta_list = [1, 5, 10, 20, 40, 80, 100, 150, 300, 500, 1000, 2000, 10000]

def bucketize(ctrs_0, weights_0, ctrs_1, weights_1, bucket_size=10):
    n_experiments, n_users = ctrs_0.shape
    n_buckets = math.ceil(n_users / bucket_size)
    values_0 = np.zeros((n_experiments, n_buckets))
    values_1 = np.zeros((n_experiments, n_buckets))
    for b in np.arange(n_buckets):
        ind = np.arange(b * n_users / n_buckets, b * n_users / n_buckets + n_users / n_buckets).astype(np.int)
        values_0[:, b] = np.sum(ctrs_0[:, ind] * weights_0[:, ind], axis=1) / np.sum(weights_0[:, ind], axis=1)
        values_1[:, b] = np.sum(ctrs_1[:, ind] * weights_1[:, ind], axis=1) / np.sum(weights_1[:, ind], axis=1)
    return values_0, values_1

def t_test_buckets(ctrs_0, weights_0, ctrs_1, weights_1, bucket_size=10):
    return t_test(*bucketize(ctrs_0, weights_0, ctrs_1, weights_1, bucket_size))

def mannwhitney_buckets(ctrs_0, weights_0, ctrs_1, weights_1, bucket_size=10):
    return mannwhitney(*bucketize(ctrs_0, weights_0, ctrs_1, weights_1, bucket_size))

def plot_all_buckets(N=20000, experiments=2000, mu=5, sigma2=1.3, success_rate=0.02, uplift=0.03, beta=100, bucket_size=10):
    views_a_1, views_a_2, views_b, \
    success_rate_a_1, success_rate_a_2, success_rate_b, \
    clicks_a_1, clicks_a_2, clicks_b = generate(N=N, experiments=experiments, mu=mu, sigma2=sigma2, success_rate=success_rate, uplift=uplift, beta=beta)
    titles = [
        't-test',
        'Mann-Whitney',
        't-test buckets',
        'Mann-Whitney buckets'
    ]

    ctrs_a_1 = clicks_a_1.astype(np.float) / views_a_1
    ctrs_a_2 = clicks_a_2.astype(np.float) / views_a_2
    ctrs_b = clicks_b.astype(np.float) / views_b

    # debug print
    print(success_rate_a_1[0])

    p_values_ab = [
        t_test(ctrs_a_1, ctrs_b),
        mannwhitney(ctrs_a_1, ctrs_b),
        t_test_buckets(ctrs_a_1, views_a_1, ctrs_b, views_b, bucket_size),
        mannwhitney_buckets(ctrs_a_1, views_a_1, ctrs_b, views_b, bucket_size)
    ]

    p_values_aa = [
        t_test(ctrs_a_1, ctrs_a_2),
        mannwhitney(ctrs_a_1, ctrs_a_2),
        t_test_buckets(ctrs_a_1, views_a_1, ctrs_a_2, views_a_2, bucket_size),
        mannwhitney_buckets(ctrs_a_1, views_a_1, ctrs_a_2, views_a_2, bucket_size)
    ]

    views_target = views_b
    ground_truth_ctr_target = success_rate_b 
    color = colors[0]

    test_data = {}
    for i,j in enumerate(titles):
        test_data[j] = (p_values_ab[i],p_values_aa[i],colors[i])
    pict = plot_summary(test_data, ground_truth_ctr_target, views_target, sigma2, beta, mu, N)
    name = f'experimens={experiments}_N={N}_mu={mu}_uplift={uplift}_success_rate={success_rate}_beta={beta}_sigma2={sigma2}_bucket_size={bucket_size}.png'
    pict.savefig(f'{name}')
    return f'{name}'

# gif_list = [plot_all_buckets(sigma2 = sigma2) for sigma2 in sigma_list]
# to_gif(gif_list, 'all_sigma_variation_bucketing.gif', delete_after=False)

# gif_list = [plot_all_buckets(beta = beta) for beta in beta_list]
# to_gif(gif_list, 'all_beta_variation_bucketing.gif', delete_after=False)

# example of how to make bucketing significantly better than the others
# gif_list = [
    # plot_all_buckets(N=20000, experiments=500, mu=5, sigma2=4.5, success_rate=0.02, uplift=0.03, beta=1000, bucket_size=10),
    # plot_all_buckets(N=50000, experiments=500, mu=1, sigma2=4.5, success_rate=0.02, uplift=0.03, beta=1000, bucket_size=10),
    # plot_all_buckets(N=20000, experiments=500, mu=1, sigma2=4.5, success_rate=0.02, uplift=0.03, beta=1000, bucket_size=10),
# ]
# to_gif(gif_list, 'all_bucketing.gif', delete_after=False)

