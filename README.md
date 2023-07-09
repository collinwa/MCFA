# Multiset Correlation and Factor Analysis (MCFA)

MCFA is a method for analysis of multimodal, high-dimensional
data that combines principals from multiset canonical correlation analysis and
factor analysis to jointly model shared and private features across the datasets.
It was designed with multi-omic analysis in mind, but may be useful for any dataset
where two or more data types are gathered for the same samples. MPCFA is implemented
in Python 3 using pytorch on the backend for speed and GPU compatibility.

## Installation and requirements

MCFA is available on PyPI and installable via `pip`. Just run

```
pip install multicor_fa
```

Dependencies are listed in `pyproject.toml` and should be automatically installed.
These include `torch`, `numpy`, `pandas`, `scipy`, and `sklearn`.

## Usage instructions and examples
### Basic model fitting
The primary interface to the fitting routines is the function `mcfa_model.fit()` which
takes a list of pandas DataFrames with equal numbers of rows (samples by features) and a set
of analysis options. By default MPCFA learns the relevant hyperparameters from the
data and thus the default options will suffice for many cases.
This interface is intended to be used via a [Jupyter](https://jupyter.org/) or
[Google Colab](colab.research.google.com/) notebook.

Given data in the matrices `Y_1, Y_2, Y_3`, the primary input to MCFA is an iterable
containing the datasets. If it is a named iterable (dictionary), those names
will be used in the returned results for clarity.

```
from multicor_fa import mcfa_model

Y = {"Mode1": Y_1, "Mode2": Y_2, "Mode3": Y_3}
mcfa_res = mcfa_model.fit(Y)
```

Calling `mcfa_model.fit(Y)` will calculate the dataset PCs, infer the hyperparameters, and fit
the model. If you'd prefer to set these yourself, you can use the argument `n_pcs` to pass
a list of number of components of each dataset to keep, `d` to pass an integer specifying
the shared dimensionality, and `k` to pass a list specifying the private space dimensionality.

```
mcfa_res = mcfa_model.fit(Y, n_pcs = [24, 11, 57], d = 5, k = [12, 3, 22])
```

The boolean arguments `center` and `scale` (default `True`) will standardize the data
if it is not already standardized. This is generally recommended and simplifies interpretaiton
of the variance explained.

If you don't need to do analysis of the factor loadings matrix, you can set the parameter
`result_space = 'pc'` which avoids transforming the loadings matrix back into the full data
space. This can save some time and memory if you have very large hidden dimensionalities and
feature spaces for your input matrices.

`mcfa_model.fit()` returns an `MCFARes` object that can be used in downstream analysis.
`mcfa_res.Z` contains a `samples x d` `pd.DataFrame` with the location of the points in
the shared space, `mcfa_res.X` contains an iterable of `samples x k_i` `pd.DataFrame`'s
with the corresponding private spaces. `W` is likewise an iterable of shared space
factor loadings dataframes `d x p_i`, and `L` is an iterable of private space
factor loadings matrices `k_i x p_i`. `var_exp_Z` and `var_exp_X` contain the
variance in each dataset explained by the shared and private spaces, respectively.


See the dosctring of the function `mcfa_model.fit()` for complete details, and
`analysis_notebooks/analyze_mesa.ipynb` to see how we used the functionality of
this package in our analysis of the MESA/TOPMed multi-omic pilot data.

### Cross Validation
Once you have fit your model, you may consider doing cross-validation to assure that you
have not over-fit. The function `mcfa_model.cv()` provides this functionality.

```
Z_cv, X_cv, nrmse_tr, nrmse_te = mcfa_model.cv(Y, mcfa_res, folds = 'loo')
```

`Z_cv` and `X_cv` will contain a `pd.DataFrame` and iterable of such where each sample
in the DataFrame corresponds to the location of that sample in the respective space
in the model fit *without* that sample included. The `nrmse_tr` and `nrmse_te` contain
normalized root mean square error for each fold in each dataset. The `folds` argument
specifies the number of folds, or use `folds='loo'` for leave-one-out cross-validation.

Keep in mind that proper choice of the hidden dimensionality is directly related to the
sample size - with fewer samples, you have to model fewer hidden dimensions. Thus, if
you use too few folds, you may observe over-fitting just because each fold doesn't have
enough samples to model the same number of hidden dimensions as the full dataset.
In the manuscript we use leave-one-out to get around this, but you can probably get away
with less.

### Feature set enrichment analysis
Generally speaking, we recommend analying variance-normalized feature loadings matrices
or Z-transformed correlations of dataset features with the hidden space of interest
to find highly-weighted features in downstream analysis. This is generally less-prone
to mis-specification than "set"-type enrichment analyses. The primary reason
for this is that the feature loadings matrix is correlated, and dealing with this
correlation is not straightforward.

You can use the function `mcfa_model.score` to calculate correlations between data features
inferred dimensions

```
cor1 = mcfa_model.score(Y['Y_1'], mcfa_res.Z, transfrom = True
```

This returns a `features x factors` `pd.DataFrame` with entries the corelation between
the feature and factor. With `transform = True` these will use Fisher's Z-transformation
to turn these into roughly normally-distributed Z-scores.

However, we do also provide functionality for
gene set enrichment analysis using the `gsea.parametric` and `gsea.permutation`
functions. These functions implement the proposed approach in Frost et al. *BioDataMining*
2015.

The simplest and most problematic approach is parametric -
```
from multicor_fa import gsea
scores, pvals = gsea.parametric(mcfa_res.W['Y_1'], annot, ref_sample = None)
```

This will compute a score and p-value for each annotation in `annot` against each dimesion
in the factor loadings matrix for dataset `Y_1`. Here, `annot` is a dictionary mapping
annotation names to feature names. You can use the function `gsea.load_annot()` to load
a GMT annotation file into the proper format. `ref_sample` gives a reference sample for
correcting for the correlation present in the factor loadings matrix. `ref_sample = None` will
perform no correction, and will likely result in many false positives. On the other hand,
you could use in-sample correction `ref_sample = Y['Y_1']`, which will likely result
in very low power. An independent reference panel may provide better correction.
If you have one, you can also specify that here.

The other approach is to do permutations -
```
scores, pvals = gsea.permutation(Y['Y_1'], mcfa_res.Z, annot, n_perm=100000, threads=10)
```

This looks for (transformed) correlations of the input data (`Y['Y_1']`) with the
reduced space (`mcfa_res.Z`) and uses permutations to correct for the correlations
across features. This generally works quite well, with the trouble being that you
need to do a *lot* of permutations. For a standard gene set enrichment analysis, you
likely need to do at least 100k permutations to get somewhat accurate p-value estimates.
If you have multiple cores available, you can speed things up with the `threads` argument.

Finally, you may be interested in doing feature loadings analysis on sets of less-interpretable
features, such as methylation markers. For this, you can use the funcion `gsea.gene_score()`.
```
scores = gsea.gene_score(mcfa_res.W['Y_2'], mapping, method = 'mean')
```

Where mapping is a dictionary mapping gene names to columns of `W`. This calculates the
mean (or optionally `method = meansq`) value of the features and returns a `pd.DataFrame`
of gene scores.

### Other notes

Please note that MCFA does no preprocessing or data input checking (eg for sample alignment)
other than optional feature centering and scaling at this time.
All additional preprocessing such as normalization, filtering of problematic samples,
imputation of missing values, and alignment of samples across datasets needs to be done
prior to calling `mcfa_model.fit()`. For an example of preprocessing steps that were taken
for the MESA cohort, see `MCFA/analysis_notebooks/preprocess_mesa.ipynb`

This software is under active development. Please report any issues
encountered using the issues tab. For the exact code used in the manuscript
"Multiset correlation and factor analysis enables exploration of multi-omic data",
Brown, Wang et al *Cell Genomics* forthcoming, see the tagged first release. An archive
of that repository is also available on Zenodo - https://doi.org/10.5281/zenodo.7951370.
