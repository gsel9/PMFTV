# LongiMC - Low-rank matrix completion for longitudinal data

# Installation

To install `longimc`, you can run

```python
python -m pip install longimc
```

# About

This Python library that adds support for low-rank matrix completion of longitudinal data. 

$$
\min_{\substack{\mathbf{U}, \mathbf{V}}}  \left \| \mathbf{W} \odot \left ( \mathbf{Y} - \mathbf{U} \mathbf{V}^\top \right ) \right \|_F^2 + \alpha_1 \left \| \mathbf{U} \right \|_F^2 + \alpha_2 \left \| \mathbf{V} \right \|_F^2 + \alpha_3 \left \| \mathbf{R} \mathbf{V} \right \|_F^2 
$$

# Example

TODO


References
----------

* [1]: Langberg, Geir Severin RE, et al. "Matrix factorization for the reconstruction of cervical cancer screening histories and prediction of future screening results." BMC bioinformatics 23.12 (2022): 1-15.
* [2]: Langberg, Geir Severin RE, et al. "Towards a data-driven system for personalized cervical cancer risk stratification." Scientific Reports 12.1 (2022): 12083.
* [2]: Elvatun, Severin, et al. Cross-population evaluation of cervical cancer risk prediction algorithms. To appear in IJMEDI (2023).
