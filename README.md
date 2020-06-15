# Clustering by using "snakes and dragons"

Implementation of the paper [Clustering of the structures by using "snakes & dragons" approach, or correlation matrix as a signal](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223267)

## Datasets

1. Macroeconomics development indicators from the World Bank - [Link](https://datacatalog.worldbank.org/dataset/world-development-indicators)

## Requirements

- NumPy
- Pandas
- Scikit-learn
- Tqdm (for displaying a progress bar)
- [Yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) (provides mechanism for selecting the best number of clusters `k`, as described in the paper)
  
  To install using the conda package mamager (recommended):

  ```bash
  conda install -c districtdatalabs yellowbrick
  ```

### Optional requirements

The algorithm internally uses `KMeans` multiple times on random partitions of the entire dataset. Although sklearn's implementation of K-Means is widely used, it is not the fastest out there. [Intel-backed DAAL's](https://github.com/IntelPython/daal4py) implementation was found to be much faster in the [initial benchmarks](https://github.com/MrinalJain17/drake/wiki/Benckmarks), giving almost 8-12x speed-up. If DAAL is **not** installed, then the code will fallback to use the sklearn's implementation.

The recommended way to install [DAAL for python](https://github.com/IntelPython/daal4py) would be using the conda package manager:

```bash
conda install -c intel daal4py
```

## Refrences

1. Consensus Clustering (paper): https://link.springer.com/article/10.1023/A:1023949509487
2. Consensus Clustering (blog): https://towardsdatascience.com/consensus-clustering-f5d25c98eaf2
