# Longitudinal Matrix Completion

The objective is

$$
\begin{align}
& F = \|P_\Omega (X - UV^\top)\|_F^2
\\
& R = \alpha_1 \| U \|_F^2 + \alpha_2 \| V \|_F^2 + \alpha_3 \| RV \|_F^2
\end{align}
$$

Here, $\| \cdot \|_F$ denotes the Frobenius norm. In the discrepancy term, $P_{\Omega}$ is the projection onto the obsered entries in $X$. The regularization $|| U ||_F^2 + || V ||_F^2$ controls overfitting, and $|| RV ||_F^2$ impose temporal smoothness on the time-varying basic profiles $V$ by penalizing rapid changes in the profiles uniformly in time. Here $R$ is a forward finite-difference matrix.
