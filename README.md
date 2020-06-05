# TCR-classifiers estimation pipeline

## Our goal
The analysis of T-cell receptors sequences of an individual allows us to determine, which pathogens and viruses his immune system can detect. The main challenge here is the uncertain matching between receptors and antigens. The TCR-sequences of different lengths and amino acids-content can be complementary to the same antigen.
The goal of this project is to develop an efficient pipeline to compare the existing methods of the TCR-corresponding antigens prediction.
We also hypothesize, that building and using custom substitution matrices from TCRs sequences, instead of BLOSUM, can improve antigen-prediction performance.

## Database
We test all of the methods on the [VDJ-database](https://vdjdb.cdr3.net/) (a curated database of T-cell receptor sequences of known antigen specificity).

## Supported algorithms
For the current moment, we support these algorithms:
1) [GLIPH](https://github.com/immunoengineer/gliph)
2) [TCRdist](https://github.com/kmayerb/tcrdist2)
3) [netTCR](https://www.biorxiv.org/content/10.1101/433706v1)
4) [pMTnet](https://github.com/tianshilu/pMTnet)

Pipelines of these algorithms are provided as Jupyter-notebooks and are placed in the folders, corresponding to their names.

## Substitution matrices
Except for the BLOSUM62 substitution matrix, we provide also substitution matrices, which are based on the TCRs receptors sequences (both for alpha-and beta-chains, and single antigens). The main pipeline of building matrices locates [here](https://github.com/antigenomics/vdjdb-classifier-benchmark/tree/master/cdr3_substitutions/CDR3_Substitutions.ipynb) in Jupyter format.
The precalculated matrices are placed [here](https://github.com/antigenomics/vdjdb-classifier-benchmark/tree/master/cdr3_substitutions/matrices).
