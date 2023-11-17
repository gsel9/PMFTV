"""A usecase demonstrating inductive matrix completion.
"""

import numpy as np 


def induction(V, X):
    """_summary_

    Args:
        V (_type_): _description_
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    U_star = (2 * V.T @ X) @ np.linalg.inv(V.T @ V)
    
    return U_star @ V.T 