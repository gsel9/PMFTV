"""NB: More bells and whistels may call for adjusted regularisation!!!

Experiments:
* Sanity check.
* Trainable theta. Can replace MAP by softmax layer? More efficient pred procedure.
* Diff init of U and V.
* Loss functions.
* Optimiser AdamW and superconvergence.
* Energy difference on edges from col graph vec signals.
* Deeper/wider model. Multigraph. Measuring and Relieving the Over-smoothing Problem for Graph NeuralNetworks from the Topological View
* Hypergraph manifold regularization. Can GDL learn hypergraphs? HOW POWERFUL ARE GRAPH NEURAL NET WORKS?
* Sketching.
"""

import numpy as np
import tensorflow as tf

from .gdl_layers import SpectralGraphConv, DiffusionLTSM


SEED = 42
DTYPE = tf.float32

tf.random.set_seed(SEED)


class MGCNN(tf.keras.Model):
    
    def __init__(self, X, diffusion_steps, Lr, Lc, rank, domain, n_conv_feat=32, ord_row_conv=5, ord_col_conv=5,
                 name="GDL", **kwargs):
        
        super(MGCNN, self).__init__(name=name, **kwargs)
        
        self.X = X
        self.Lr = Lr
        self.Lc = Lc
        self.rank = rank
        self.shift = domain[0]
        self.scale = domain[-1] - domain[0]
        self.n_conv_feat = n_conv_feat
        self.ord_row_conv = ord_row_conv
        self.ord_col_conv = ord_col_conv
        self.diffusion_steps = diffusion_steps

        self.n_iter_ = 0

        self.prep_tensors()
        self.init_trainable_layers()
    
    def prep_tensors(self):
           
        # Normalised Laplacians are used in Chebyshev polynomials.
        self.norm_Lr = self.Lr - tf.linalg.tensor_diag(tf.ones([self.Lr.shape[0],]))
        self.norm_Lc = self.Lc - tf.linalg.tensor_diag(tf.ones([self.Lc.shape[0],]))

    def init_trainable_layers(self):
        
        self.convU = SpectralGraphConv(self.rank,
                                       n_conv_feat=self.n_conv_feat,
                                       ord_conv=self.ord_row_conv,
                                       seed=SEED, name='convU')
            
        self.convV = SpectralGraphConv(self.rank,
                                       n_conv_feat=self.n_conv_feat,
                                       ord_conv=self.ord_col_conv,
                                       seed=SEED, name='convV')

        self.convU.compute_cheb_polynomials(self.norm_Lr)
        self.convV.compute_cheb_polynomials(self.norm_Lc)

        self.diffU = DiffusionLTSM(units=self.X.shape[0],
                                   n_conv_feat=self.n_conv_feat,
                                   output_dim=self.rank,
                                   seed=SEED, name='LSTMU')

        self.diffV = DiffusionLTSM(units=self.X.shape[1],
                                   n_conv_feat=self.n_conv_feat,
                                   output_dim=self.rank,
                                   seed=SEED, name='LSTMV')

    def get_config(self):
       
        config = {}
        for layer in self.layers:
            config[layer.name] = layer.get_config()

        return config
    
    # TODO: 
    # * Use self.norm_X as samples from M distribution and implement Bayesian scheme. Make theta-parameter trainable.
    # * Try also linear scheme. 
    # * Is there a fast Softmax version to obtain probabilities?
    # * Change relation Y to M by a distribution emphasising normals and high-grade less so. Let parameters be trainable. 
    # NB: Check scheme is differentiable! Is more capable of learning such a parameter? Investigate ability in isolated 
    # experiments. Connection to GANS.

    # TODO: How about convolution after LSTM step and then another LSTM step to make deeper model?
    # Skip-connections to prevent over-smoothing?
    def call(self, inputs):
        
        # NB: Reset hidden and carrier state for each epoch.
        self.diffU.init_non_trainable_weights()
        self.diffV.init_non_trainable_weights()

        # NB: Reset components for each epoch.
        U_ = tf.cast(inputs[0], dtype=DTYPE)
        V_ = tf.cast(inputs[1], dtype=DTYPE)

        for _ in range(self.diffusion_steps):
            U_ += self.diffU(self.convU(U_))
            V_ += self.diffV(self.convV(V_))

        self.n_iter_ += 1

        return [U_, V_]