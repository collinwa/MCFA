# pylint: disable=invalid-name
"""Gene set enrichment funtions for MCFA

Contains functions for using MCFA results to conduct gene set enrichment
analyses.

Usage:
  TODO(brielin): Add usage example
"""

import functools
import numpy as np
import pandas as pd
import time
from concurrent import futures
from multicor_fa import mcfa_model
from scipy import stats


def load_annot(gmt_file):
    """Loads a GMT annotation file into a dict mapping annot to gene list."""
    annot = {}
    with open(gmt_file) as f:
        for line in f:
            line_split = line.split()
            annot[line_split[0]]  = line_split[2:]
    return annot


def gene_score(W: pd.DataFrame, mapping: dict, method: str = 'mean',
               min_features: int = 5):
    """Creates gene scores for non-gene-centric data.

    Args:
      W: Weight matrix (features x factors). Rows will be converted
        to gene scores via method.
      mapping: A dictionary mapping gene names to rows of W.
      method: Approach for gene score calculation. 'mean' or
        'meansq' for mean of squares.
      min_features: Minimum number of features to calculate score.
    """
    if method not in ['mean', 'meansq']:
        raise NotImplementedError
    scores = {}
    for gene, features in mapping.items():
        avail_features = W.index.intersection(features)
        if len(avail_features) > min_features:
            if method == 'mean':
                score = W.loc[avail_features].mean()
            elif method == 'meansq':
                score = (W.loc[avail_features]**2).mean()
            scores[gene] = score
    return pd.DataFrame(scores).T


def parametric(W: pd.DataFrame, annot: dict,
               ref_sample: pd.DataFrame = None, sign = True,
               min_genes = 5, shrink = 0):
    """Performs gene set enrichment analysis (GSEA).

    Peforms GSEA using the parametric PCGSE approach of Frost 2015 BDM.

    Args:
      W: A p (features) by k (factors) DataFrame with index corresponding
        to the values in annot.
      annot: A dictionary of gene set annotations mapping annotation
        names to a list of gene ids.
      ref_sample: An N (samples) by p (features) DataFrame with reference
        samples to use as feature correlations
      sign: True to consider the sign of W, otherwise compute test statistics
        using abs(W).
      min_genes: The minimum number of genes in an annotation to include.
    Returns:
      A tuple of len(annot) x k DataFrames, the first containing the
      enrichment test statistics as entries, the second containing p-values.
    """
    p = W.shape[0]
    sigma_p = np.sqrt(W.var())
    score_df = {}
    pv_df = {}
    for category, gene_set in annot.items():
        if ref_sample is not None:
            genes_in_set = ref_sample.columns.intersection(gene_set)
            genes_not_in_set = ref_sample.columns.difference(gene_set)
            m_k = len(genes_in_set)
            if m_k < min_genes:
                continue
            rho_k = (np.triu(ref_sample[genes_in_set].corr()*(1-shrink), k=1).sum()) / m_k
            df = len(ref_sample.index) - 2
        else:
            genes_in_set = W.index.intersection(gene_set)
            genes_not_in_set = W.index.difference(gene_set)
            m_k = len(genes_in_set)
            if m_k < min_genes:
                continue
            rho_k = 0
            df = p - 2
        VIF = 1 + rho_k
        if sign:
            in_set_mean = W.loc[genes_in_set].mean()
            out_set_mean = W.loc[genes_not_in_set].mean()
        else:
            in_set_mean = abs(W).loc[genes_in_set].mean()
            out_set_mean = abs(W).loc[genes_not_in_set].mean()
        num = (in_set_mean - out_set_mean)
        denom = (sigma_p * np.sqrt( VIF / m_k + 1 / (p - m_k)))

        Z_k = num/denom
        score_df[category] = Z_k
        if sign:
            pv_df[category] = 2 * (1 - stats.t.cdf(abs(Z_k), df=df))
        else:
            pv_df[category] = 1 - stats.t.cdf(Z_k, df=df)
    return pd.DataFrame(score_df).T, pd.DataFrame(pv_df).T


def _gsea_one_perm(it, start_time, data, Z, transform, annot, min_genes, sign):
    if it%100 == 0:
        print('Permutation {0:d}. Time {1:.2f}s'.format(it, time.time()-start_time))
    perm_data = data.sample(frac=1)
    perm_scores = mcfa_model.score(perm_data, Z, transform)
    perm_stats = parametric(
        W=perm_scores, annot=annot, ref_sample=None,
        sign=sign, min_genes=min_genes, shrink=0)[0].values
    return perm_stats


def permutation(data: pd.DataFrame, Z: pd.DataFrame,
                annot: dict, n_perm = 1000, sign = True, min_genes = 5,
                transform = True, threads = 1):
    """Performs GSEA using the permutation approach of Frost 2015.

    Args:
      data: A pandas DataFrame with column names matching entires in annot.
        The original data.
      Z: The feature set used to calculate gene statistics. Usually mcfa_res.Z.
      annot: A dictionary of gene set annotations mapping annotation
        names to a list of gene ids.
      n_perm: Number of permutations.
      sign: True to consider the sign of W, otherwise compute test statistics
        using abs(W).
      min_genes: The minimum number of genes in an annotation to include.
      transform: bool, true to transform correlation coefficients to Z-scores.
    Returns:
      A tuple of len(annot) x k DataFrames, the first containing the
      enrichment test statistics as entries, the second containing p-values.
    """
    start = time.time()
    f = functools.partial(
        _gsea_one_perm, start_time=start, data=data, Z=Z,
        transform=transform, annot=annot, min_genes=min_genes, sign=sign)
    with futures.ProcessPoolExecutor(max_workers=threads) as executor:
        all_perm_stats = executor.map(f, range(n_perm))
    perm_stats = np.stack(all_perm_stats)
    print('Finished in {0:f}'.format(time.time()-start))
    true_scores = mcfa_model.score(data, Z, transform)
    true_stats, _ = parametric(W=true_scores, annot=annot, ref_sample=None,
                               sign=sign, min_genes=min_genes, shrink=0)
    print(perm_stats.shape)
    print(true_stats.shape)
    if sign:
        p_vals = 1 - np.sum(true_stats.values**2 > perm_stats**2, 0) / n_perm
    else:
        p_vals = 1 - np.sum(true_stats.values > perm_stats, 0) / n_perm
    p_vals = pd.DataFrame(p_vals, index=true_stats.index)
    return true_stats, p_vals
