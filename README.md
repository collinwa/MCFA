# Multi-set Probabilistic Correlation and Factor? Analysis (MPCFA?)

MPCFA (working title) is a method for analysis of multimodal, high-dimensional
data that combines principals from multiset canonical correlation analysis and
factor analysis to jointly model shared and private features across the datasets.
It was designed with multi-omic analysis in mind, but may be useful for any dataset
where two or more data types are gathered for the same samples. MPCFA is implemented
in Python 3 using pytorch on the backend for speed and GPU compatibility.

## Installation and requirements

This software is pre-release and no specific installation instructions are provided.
The necessary python source files are in `MPCCA/py/`. The requirements can be
deduced from the first lines, and you should be able to import the methods by
placing the Python source files in the appropriate directory on your machine.

## Usage instructions and examples

The primary interface to the fitting routines is the function `mpcca.mpcca()` which
takes a list of Tensors with equal numbers of rows (samples by features) and a set
of analysis options. By default MPCFA learns the relevant hyperparameters from the
data and thus the default options will suffice for many cases, but see the documentation
of the function `mpcca.mpcca()` for complete details. For an example of this method
applied to the MESA cohort, see the IPython notebook at
`MPCCA/analysis_notebooks/analyze_mesa.ipynb`.

Please note that MPCFA does NO preprocessing or data input checking (eg for sample alignment)
at this time. ALL preprocessing such as normalization, filtering of problematic samples,
imputation of missing values, and alignment of samples across datasets needs to be done
prior to calling `mpcca.mpcca()`. For an example of preprocessing steps that were taken
for the MESA cohort, see `MPCCA/analysis_notebooks/preprocess_mesa.ipynb`
