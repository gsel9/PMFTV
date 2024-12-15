# Least-angle regression MC (LarsMC):

The least-angle regression (Lars) MC fits an L1 prior as regularizer of the coefficient matrix $U$. This will encourage sparisty in the profile coefficients

$$
\begin{align}
& F = ||P_\Omega (X - UV^\top)||_F^2
\\
& R = \alpha_1 || U ||_1 + \alpha_2 || V ||_F^2 + \alpha_3 || CRV ||_F^2
\end{align}
$$

Promoting sparsity in $U$ can be useful when choosing the rank parameter $r$ for the factorization model.
