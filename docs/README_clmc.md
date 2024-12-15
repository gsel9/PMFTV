# Convolutional Matrix Completion (CMC)

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

# Weighted Convolutional Matrix Completion

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
