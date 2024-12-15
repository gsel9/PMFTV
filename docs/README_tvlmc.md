# Total variation MC (TVMC):

The TVMC uses total variation regularization of the basic profiles to promote a more piece-wise continous solution for the temporal relationship. This regularization is useful when the profiles are changing more rapidly is not smooth over time.

$$
\begin{align}
& F = ||P_\Omega (X - UV^\top)||_F^2
\\
& R = \alpha_1 || U ||_F^2 + \alpha_2 || V ||_F^2 + \alpha_3 || \nabla V ||_1
\end{align}
$$

Compared to using $|| RV ||_F^2$ in LMC and $|| CRV ||_F^2$ in CMC, using the term $|| \nabla V ||_1$ yields a more piece-wise constant solution.
