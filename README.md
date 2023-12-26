[Installation](#Installation) | [Usage](#Usage) | [About](#About) | [Examples](#Examples) | [License](#License) | [References](#References) 

# LMC - Low-rank matrix completion for longitudinal data

![GitHub CI](https://github.com/gsel9/dgufs/actions/workflows/ci.yml/badge.svg)
![GitHub CI](https://img.shields.io/badge/code%20style-black-000000.svg)

[Matrix completion](https://en.wikipedia.org/wiki/Matrix_completion) is about fill in the missing entries of a matrix. An example of such a matrix is from the [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize) where the movie ratings from each used was organised as scarce vectors fitted into a matrix. If the data is longitudinal, the goal is to fill in the missing entries of a matrix where the columns represents time stamps.

# Installation

To install `lmc`, you can run

```python
python -m pip install lmc
```

# Usage_ 

```python
# local
from lmc import CMC
from utils import train_test_data

# third party
from sklearn.metrics import mean_squared_error

X, O_train, O_test = train_test_data()

X_train = X * O_train
X_test = X * O_test

model = CMC(rank=5, n_iter=103)
model.fit(X_train)

Y_test = model.M * O_test

score = mean_squared_error(X_test, Y_test)
```

# About

This Python library that adds support for low-rank matrix completion of longitudinal data. Algorithms based on [1], [2] and [3] are implemented as functionality for longitudinal matrix completion. 

**Data**: The data is assumed to be a partially observed data matrix $X \in \mathbb{R}^{N \times T}$. Each row $1 \leq n \leq N$ of $X$ is assumed to be a partially observed longitudinal profile

**The low-rank matrix factorization model**: The basic model for the data is that the matrix $X$ can be decomposed into a a set of shared basic profiles $\mathbf{v}_1, \dots, \mathbf{v}_r$ with
$r \ll \min \{N,T\}$. in $V$ and profile-specific coefficients in $U$.

* The observed entries $(n, t) \in \Omega$ of $X$ are possibly inaccurate measurements of a continuous \emph{latent state} $M_{n,t}$ that evolves slowly over time.
* Furthermore, we assume that each latent profile is a linear combination of a small number of *basic profiles* 

Then the matrix $\textbf{M}$ of all such profiles can be approximately decomposed as $\textbf{M} \approx \mathbf{U}\mathbf{V}^\top$ with $\mathbf{V} \in \mathbb{R}^{T \times r}$ being the collection of basic profiles
and $\mathbf{U} \in \mathbb{R}^{N\times r}$ being the row-specific coefficients. TODO: Figure \ref{fig:factoring} illustrates the latent state %risk 
model. 

**Optimization**:

The general objective is on the form
$$\min_{\substack{\mathbf{U}, \mathbf{V}}} F(\mathbf{U}, \mathbf{V}) + R(\mathbf{U}, \mathbf{V})$$
The term $F$ denotes a convex data discrepancy term and $R$ is a possibly non-smooth or possibly non-convex regularization term. 

The optimization algorithm is based on alternating minimization.

**Matrix completion models**:

Specific implementations are listed in the following.

## Longitudinal matrix completion (LMC) 

Assuem temporal smoothness and impose a constraint on the time-varying basic profiles $V$ via the regualrizationerm $|| RV ||_F^2$. Here $R$ is a forward finite-difference matrix. 

## Convolutional matrix completion (CMC) 

$$F = ||P_\Omega (X - UV^\top)||_F^2$$

$$R = || U ||_F^2 + || V ||_F^2 + || CRV ||_F^2$$

Here, $R$ is a forward difference matrix and $C$ is a the Toeplitz matrix with entries
$C_{i, j} = \exp(- \gamma \lvert i-j\rvert)$. This leads to a weaker penalisation of the profiles at faster scales and consequently allows for a larger local variability.

## Weighted CMC (WCMC)

The WCMC differs from CMC in that the projection onto the observed set is replaced by a weight matrix
$$||W \odot (S - UV^\top)||_F^2$$

Here, $W \in \mathbb{R}^{N \times T}$ sets all matrix entries $(\tilde{n}, \tilde{t}) \notin \Omega$ to $0$ and multiplies the error over the predicted values at the observed entries $(n, t) \in \Omega$ with some weights $W_{n, t} > 0$. These weights provide a flexible way to incorporate additional information such as uncertainties in the observed entries results and adjusting for entries $Y_{n, t}$ not missing at random with inverse propensity weighting [4].

$$R = || U ||_F^2 + || V ||_F^2 + || CRV ||_F^2$$

One solution uses gradient descent. The other uses ADMM and involves an extra parameter $\beta$.

## Shifted CMC (SCMC)

$$F =  ||W \odot (X - UV^\top Z)||_F^2$$

$$R = || U ||_F^2 + || V ||_F^2 + || CRV ||_F^2$$

Shifts can be (1) integer shifts; (2) continous in time.

## Total variation MC (TVMC):

$$F = ||P_\Omega (X - UV^\top)||_F^2$$

$$R = || U ||_F^2 + || V ||_F^2 + || \nabla V ||_1$$

## Least-angle regression MC (LarsMC):

The least-angle regression (Lars) MC fits an L1 prior as regularizer of the coefficient matrix $U$. The optimization objective for LarsMC consists of 

$$F = ||P_\Omega (X - UV^\top)||_F^2$$

$$R = || U ||_1 + || V ||_F^2 + || CRV ||_F^2$$

# Examples

Usecases 

## Synthetic control method (SCM)

Imbalanced data, WCMC

## Inductive matrix completion

## Rank selection

LarsMC

## Latent variable model

MAP predict, CMC

## Phase shifted data

DGD profiles, SCMC

# License 

`lmc` was created by Severin Elvatun. It is licensed under the terms of the MIT license.

# References

* [1]: Langberg, Geir Severin RE, et al. "Matrix factorization for the reconstruction of cervical cancer screening histories and prediction of future screening results." BMC bioinformatics 23.12 (2022): 1-15.
* [2]: Langberg, Geir Severin RE, et al. "Towards a data-driven system for personalized cervical cancer risk stratification." Scientific Reports 12.1 (2022): 12083.
* [3]: Elvatun, Severin, et al. Cross-population evaluation of cervical cancer risk prediction algorithms. To appear in IJMEDI (2023).
* [4]: Schnabel, Tobias, et al. "Recommendations as treatments: Debiasing learning and evaluation." international conference on machine learning. PMLR, 2016.
