[Installation](#Installation) | [Usage](#Usage) | [About](#About) | [Examples](#Examples) | [License](#License) | [References](#References) 

# LMC 
Low-rank matrix completion for longitudinal data with various discrepancy terms and regularizations.

![GitHub CI](https://github.com/gsel9/dgufs/actions/workflows/ci.yml/badge.svg)
![GitHub CI](https://img.shields.io/badge/code%20style-black-000000.svg)

---

[Matrix completion](https://en.wikipedia.org/wiki/Matrix_completion) is about fill in the entries of a scarce matrix. An example of such a matrix is from the [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize), where the rating scores from each user for a small number of movies are organised as scarce vectors fitted into a matrix. 

Less frequently studied is when the data is longitudinal and the goal is to complete the temporal relationship between measurements. This Python library that adds support for low-rank matrix completion of longitudinal data.

# Installation

To install `lmc` from the source directory you can run 

```python
python -m pip install .
```

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

# About

This Python library implements algorithms based on [1], [2] and [3] for for low-rank matrix completion of longitudinal data.

**Data**: The data is assumed to be organized as a partially observed data matrix $X \in \mathbb{R}^{N \times T}$. Each row $1 \leq n \leq N$ of $X$ is a partially observed profile of longitudinal measurements. Each column $1 \leq t \leq T$ of $X$ represents a unit of time. Assuming the profiles in $X$ are correlated, this matrix can be approximately low rank.

**Low-rank factorization model**: The basic factorization model for the data is that the matrix $X$ can be decomposed into a set of shared time-varying basic profiles $\mathbf{v}_1, \dots, \mathbf{v}_r$ with
$r \ll \min (\{N,T\})$ and profile-specific coefficients in $U_n$. The linear combination $M_n = U_nV^\top$ of coefficients and basic profiles yields the estimate for the reconstructed data profile. Assuming the profiles in $X$ are correlated, the matrix $\textbf{M}$ of all such profiles can be approximately decomposed as $\textbf{M} \approx \mathbf{U}\mathbf{V}^\top$ with $\mathbf{V} \in \mathbb{R}^{T \times r}$ being the collection of basic profiles and $\mathbf{U} \in \mathbb{R}^{N\times r}$ being the profile-specific coefficients. The task is thus to estimate $U$ and $V$ from $X$.

**Optimization**:

The general objective to estimate $U$ and $V$ from $X$ is on the form
$$\min_{\substack{\mathbf{U}, \mathbf{V}}} F(\mathbf{U}, \mathbf{V}) + R(\mathbf{U}, \mathbf{V})$$
The various optimization algorithms are based on alternating minimization. Here, $F$ is a data discrepancy term and $R$ is regularization used to impose specific structures on the result. Depending on the expected structure of $M$, various constraints may be imposed. The specific implementations are listed in the following.

## Longitudinal Matrix Completion (LMC) 

This is the most basic implementation where 

$$
\begin{align}
& F = ||P_\Omega (X - UV^\top)||_F^2
\\ 
& R = \alpha_1 || U ||_F^2 + \alpha_2 || V ||_F^2 + \alpha_3 || RV ||_F^2
\end{align}
$$


Here, $|| \cdot ||_F$ denotes the Frobenius norm. In the discrepancy term, $P_{\Omega}$ is the projection onto the obsered entries in $X$. The regularization $|| U ||_F^2 + || V ||_F^2$ controls overfitting, and $|| RV ||_F^2$ impose temporal smoothness on the time-varying basic profiles $V$ by penalizing rapid changes in the profiles uniformly in time. Here $R$ is a forward finite-difference matrix. 

## Convolutional Matrix Completion (CMC) 

The CMC is similar to the LMC, except that it adds a convolution of the finite difference matrix for penalizing the time-varying basic profiles. This has the effect of reducing penalization of rapid changes in the time-varying basic profiles, allowing for more rapid local changes in the profiles, but it will also encourage more upstream and downstream profile smoothness. Here

$$
\begin{align}
& F = ||P_\Omega (X - UV^\top)||_F^2
\\ 
& R = \alpha_1 || U ||_F^2 + \alpha_2 || V ||_F^2 + \alpha_3 || CRV ||_F^2
\end{align}
$$

Again, $R$ is a forward difference matrix and $C$ is a the Toeplitz matrix with entries
$C_{i, j} = \exp(- \gamma \lvert i-j\rvert)$. This leads to a weaker penalisation of the profiles at faster scales and consequently allows for a larger local variability.

## Weighted Convolutional Matrix Completion (WCMC)

In the WCMC, the projection $P_{\Omega}$ onto the obsered entries in $X$ is replaced by a masked weight matrix $W \in \mathbb{R}^{N \times T}$. 

$$
\begin{align}
& F = ||W \odot (X - UV^\top)||_F^2
\\ 
& R = \alpha_1 || U ||_F^2 + \alpha_2 || V ||_F^2 + \alpha_3 || CRV ||_F^2
\end{align}
$$

The matrix $W$ sets all matrix entries $(\tilde{n}, \tilde{t}) \notin \Omega$ to $0$ and multiplies the error over the predicted values at the observed entries $(n, t) \in \Omega$ with some weights $W_{n, t} > 0$. These weights provide a flexible way to incorporate additional information such as uncertainties in the observed entries results and adjusting for entries $Y_{n, t}$ not missing at random with inverse propensity weighting [4].

Two solvers are currently implemented for this method. One is based on automatic differentiation to yield an approximate solution, while the other uses ADMM and introduces an additional tuning parameter $\beta$. Depenting on the choice of solver method, the reaults may vary.  

## Total variation MC (TVMC):

The TVMC uses total variation regularization of the basic profiles to promote a more piece-wise continous solution for the temporal relationship. This regularization is useful when the profiles are changing more rapidly is not smooth over time.

$$
\begin{align}
& F = ||P_\Omega (X - UV^\top)||_F^2
\\ 
& R = \alpha_1 || U ||_F^2 + \alpha_2 || V ||_F^2 + \alpha_3 || \nabla V ||_1
\end{align}
$$

Compared to using $|| RV ||_F^2$ in LMC and $|| CRV ||_F^2$ in CMC, using the term $|| \nabla V ||_1$ yields a more piece-wise constant solution.

## Least-angle regression MC (LarsMC):

The least-angle regression (Lars) MC fits an L1 prior as regularizer of the coefficient matrix $U$. This will encourage sparisty in the profile coefficients 

$$
\begin{align}
& F = ||P_\Omega (X - UV^\top)||_F^2
\\ 
& R = \alpha_1 || U ||_1 + \alpha_2 || V ||_F^2 + \alpha_3 || CRV ||_F^2
\end{align}
$$

Promoting sparsity in $U$ can be useful when choosing the rank parameter $r$ for the factorization model.

## Shifted CMC (SCMC)

In the SCMC, the factorization model is different from the previous ones.

$$
\begin{align}
& F = ||P_\Omega (X - UV^\top Z)||_F^2
\\ 
& R = \alpha_1 || U ||_F^2 + \alpha_2 || V ||_F^2 + \alpha_3 || CRV ||_F^2
\end{align}
$$

Three versions of basic profile shifting are implemented: discrete-time shifts, continous-time shifts based on Fourier transformation and ... 

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
