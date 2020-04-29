# Probabilistic Matrix Factorisation with Total Variation

## Goal:
Find an alternative approach to update V in PACIFIER framework that allows for TV regularisation.

## To-dos:
- GREAT IDEA: Make dir `experiments` with functinos that can be run by a `main.py` script at the root. This way can store all experiments as separate scripts while main is still at top level with access to all utilities.

## Notes
- Subderivative/subgradient/subdifferential: derivative generalise to non-differentiable convex functions.
- Proximal gradient methods for learning: convex regularization algorithms where the regularization may not be differentiable.
- For isotropic TV, the prox is a projection onto the L2 unit ball.
- For anisotropic TV, this is a projection onto the L-infinity unit ball.
- Negative divergence is adjoint of gradient.
- Adjoint of a real matrix is its transpose.
