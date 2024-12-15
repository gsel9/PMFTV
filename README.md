[Installation](#Installation) | [Usage](#Usage) | [About](#About) | [Examples](#Examples) | [License](#License) | [References](#References)

# LMC: Low-rank matrix completion for longitudinal data

![GitHub CI](https://github.com/gsel9/dgufs/actions/workflows/ci.yml/badge.svg)
![GitHub CI](https://img.shields.io/badge/code%20style-black-000000.svg)

## About

Low-rank matrix completion (LMC) is a technique used to recover a partially observed matrix by exploiting the assumption that the matrix has a low-rank structure. In LMC for longitudinal data, the data was collected over time, potentially at irregular intervals.

In this context, the matrix can be represented with rows corresponding to entities, and columns corresponding to the time points. Each entry in the matrix would represent an observation for a particular entity at a particular time. Low-rank matrix completion aims to recover the missing entries by exploiting the assumption that the matrix, when viewed as time-dependent for each entity, can be approximated by a low-rank matrix.

The key idea is that, even though there might be missing values, the relationships or patterns in the data across individuals and time points can be captured by a low-rank approximation. This allows for imputation of missing values based on the observed data, by learning both entity-specific and time-dependent trends in the data.

**Data**: The data is assumed to be organized as a partially observed data matrix $X \in \mathbb{R}^{N \times T}$. Each row $1 \leq n \leq N$ of $X$ is a partially observed profile of longitudinal measurements. Each column $1 \leq t \leq T$ of $X$ represents a unit of time. Assuming the profiles in $X$ are correlated, this matrix can be approximately low rank.

**Low-rank factorization model**: The basic factorization model for the data is that the matrix $X$ can be decomposed into a set of shared time-varying basic profiles $\mathbf{v}_1, \dots, \mathbf{v}_r$ with
$r \ll \min (\{N,T\})$ and profile-specific coefficients in $U_n$. The linear combination $M_n = U_nV^\top$ of coefficients and basic profiles yields the estimate for the reconstructed data profile. Assuming the profiles in $X$ are correlated, the matrix $\textbf{M}$ of all such profiles can be approximately decomposed as $\textbf{M} \approx \mathbf{U}\mathbf{V}^\top$ with $\mathbf{V} \in \mathbb{R}^{T \times r}$ being the collection of basic profiles and $\mathbf{U} \in \mathbb{R}^{N\times r}$ being the profile-specific coefficients. The task is thus to estimate $U$ and $V$ from $X$.

**Optimization**:

The general objective to estimate $U$ and $V$ from $X$ is on the form
$$\min_{\substack{\mathbf{U}, \mathbf{V}}} F(\mathbf{U}, \mathbf{V}) + R(\mathbf{U}, \mathbf{V})$$
The various optimization algorithms are based on alternating minimization. Here, $F$ is a data discrepancy term and $R$ is regularization used to impose specific structures on the result. Depending on the expected structure of $M$, various constraints may be imposed. The specific implementations are listed in the following.

## Key features

* [Longitudinal matrix completion](./docs/README_lmc.md)
* [Convolutional longitudinal matrix completion](./docs/README_clmc.md)
* [Total variation longitudinal matrix completion](./docs/README_tvlmc.md)
* [Least-angle regression matrix completion](./docs/README_lars.md)
* [Phase-shifted matrix completion](./docs/README_slmc.md)

## Installation

Install the library with pip:
```
pip install .
```
This ensures dependencies listed in `pyproject.toml` are handled correctly.

# Usage

A basic example involves estimating the entries of a matrix $X$, given only the entries indicated by $O_{train}$ and use the entries in $O_{test}$ to evaluate the reconstruction accuracy.

```python
# lmc lib
from lmc import CMC
from utils import train_test_data

# third party
from sklearn.metrics import mean_squared_error

X, O_train, O_test = train_test_data()

X_train = X * O_train
X_test = X * O_test

model = CMC(rank=5, n_iter=100)
model.fit(X_train)

Y_test = model.M * O_test

score = mean_squared_error(X_test, Y_test)
```
