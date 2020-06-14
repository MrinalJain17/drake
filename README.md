# Clustering by using "snakes and dragons"

Implementation of the paper [Clustering of the structures by using "snakes & dragons" approach, or correlation matrix as a signal](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223267)

## Datasets

1. Macroeconomics development indicators from the World Bank - [Link](https://datacatalog.worldbank.org/dataset/world-development-indicators)

## Requirements

- NumPy
- Pandas
- Scikit-learn
- Tqdm

The algorithm internally uses `KMeans` multiple times on random partitions of the dataset. Although sklearn's implementation of KMeans is widely used, it is not the fastest out there. [Intel-backed DAAL's](https://github.com/IntelPython/daal4py) implementation was found to be much faster in the initial benchmarks, giving almost 6-8x speed-up. If DAAL is **not** installed, then the code will fall back to use sklearn's implementation of `KMeans`.

The recommended way to install [DAAL for python](https://github.com/IntelPython/daal4py) would be using the conda package manager:

```bash
conda install -c intel daal4py
```

## Refrences

1. The code for the consensus clustering module was (initially) partially adapted from [this repository](https://github.com/ZigaSajovic/Consensus_Clustering). Later, much of it was reworked in [this commit](https://github.com/MrinalJain17/drake/commit/0a8e98a5886ac2df2265eb907111992ebc1de019).
