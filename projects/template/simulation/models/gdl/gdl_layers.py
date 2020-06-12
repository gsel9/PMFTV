"""Model layers."""

# -*- coding: utf-8 -*-

__author__ = 'Severin Langberg'

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers


SEED = 0

tf.random.set_seed(SEED)

DTYPE = tf.float32

   
class DiffusionLTSM(layers.Layer):
    """
    Args:
        units: M.shape[0] for rows and  M.shape[1] for cols.
    """
    def __init__(self, units, output_dim, n_conv_feat, seed=None, name=None):

        super().__init__(trainable=True, name=name, dtype=DTYPE)
        
        self.units = units
        self.seed = seed
        self.output_dim = output_dim
        self.n_conv_feat = n_conv_feat
    
        self.init_trainable_weights(name)
        self.init_non_trainable_weights()
    
    def init_trainable_weights(self, name):
        
        w_init = tf.keras.initializers.GlorotNormal(seed=self.seed)
        
        self.Wf = tf.Variable(initial_value=w_init(shape=(self.n_conv_feat, self.n_conv_feat),
                                                   dtype=DTYPE),
                              name=f'{name}_Wf',
                              trainable=True)

        self.Wi = tf.Variable(initial_value=w_init(shape=(self.n_conv_feat, self.n_conv_feat),
                                                   dtype=DTYPE),
                              name=f'{name}_Wi',
                              trainable=True)

        self.Wo = tf.Variable(initial_value=w_init(shape=(self.n_conv_feat, self.n_conv_feat),
                                                   dtype=DTYPE),
                              name=f'{name}_Wo',
                              trainable=True)

        self.Wc = tf.Variable(initial_value=w_init(shape=(self.n_conv_feat, self.n_conv_feat),
                                                   dtype=DTYPE),
                              name=f'{name}_Wc',
                              trainable=True)

        self.Uf = tf.Variable(initial_value=w_init(shape=(self.n_conv_feat, self.n_conv_feat),
                                                   dtype=DTYPE),
                              name=f'{name}_Wf',
                              trainable=True)

        self.Ui = tf.Variable(initial_value=w_init(shape=(self.n_conv_feat, self.n_conv_feat),
                                                   dtype=DTYPE),
                              name=f'{name}_Ui',
                              trainable=True)

        self.Uo = tf.Variable(initial_value=w_init(shape=(self.n_conv_feat, self.n_conv_feat),
                                                   dtype=DTYPE),
                              name=f'{name}_Uo',
                              trainable=True)

        self.Uc = tf.Variable(initial_value=w_init(shape=(self.n_conv_feat, self.n_conv_feat),
                                                   dtype=DTYPE),
                              name=f'{name}_Uc',
                              trainable=True)
        
        b_init = tf.zeros_initializer()
        
        self.bf = tf.Variable(initial_value=b_init(shape=(self.n_conv_feat,), dtype=DTYPE),
                              name=f'{name}_bf', trainable=True)

        self.bi = tf.Variable(initial_value=b_init(shape=(self.n_conv_feat,), dtype=DTYPE),
                              name=f'{name}_bi', trainable=True)

        self.bo = tf.Variable(initial_value=b_init(shape=(self.n_conv_feat,), dtype=DTYPE),
                              name=f'{name}_bo', trainable=True)

        self.bc = tf.Variable(initial_value=b_init(shape=(self.n_conv_feat,), dtype=DTYPE),
                              name=f'{name}_bo', trainable=True)
                              
        self.Wout = tf.Variable(initial_value=w_init(shape=(self.n_conv_feat, self.output_dim),
                                                     dtype=DTYPE),
                                name=f'{name}_Wout', trainable=True)
                                
        self.bout = tf.Variable(initial_value=b_init(shape=(self.output_dim,), dtype=DTYPE),
                                                        name=f'{name}_bout', trainable=True)
    
    def init_non_trainable_weights(self):

        init = tf.zeros_initializer()
        
        self.h = tf.zeros([self.units, self.n_conv_feat], dtype=DTYPE)
        self.c = tf.zeros([self.units, self.n_conv_feat], dtype=DTYPE)
    
    def reset_h_c(self):
        """Note: Needs to be re-initalised for each epoch."""
        self.init_non_trainable_weights()

    def call(self, inputs):
        
        f = tf.sigmoid(tf.matmul(inputs, self.Wf) + tf.matmul(self.h, self.Uf) + self.bf)
        i = tf.sigmoid(tf.matmul(inputs, self.Wi) + tf.matmul(self.h, self.Ui) + self.bi)
        o = tf.sigmoid(tf.matmul(inputs, self.Wo) + tf.matmul(self.h, self.Uo) + self.bo)
    
        self.c_update = tf.sigmoid(tf.matmul(inputs, self.Wc) + tf.matmul(self.h, self.Uc) + self.bc)
        
        self.c = tf.multiply(f, self.c) + tf.multiply(i, self.c_update)
        self.h = tf.multiply(o, tf.sigmoid(self.c))
        
        return tf.nn.tanh(tf.matmul(self.c, self.Wout) + self.bout)

    def get_config(self):

        config = super(DiffusionLTSM, self).get_config()
        config.update({'Wf': self.Wf.numpy().tolist()})
        config.update({'Wi': self.Wi.numpy().tolist()})
        config.update({'Wo': self.Wo.numpy().tolist()})
        config.update({'Wc': self.Wc.numpy().tolist()})
        config.update({'Uf': self.Uf.numpy().tolist()})
        config.update({'Ui': self.Ui.numpy().tolist()})
        config.update({'Uo': self.Uo.numpy().tolist()})
        config.update({'Uc': self.Uc.numpy().tolist()})
        config.update({'bf': self.bf.numpy().tolist()})
        config.update({'bi': self.bi.numpy().tolist()})
        config.update({'bo': self.bo.numpy().tolist()})
        config.update({'bc': self.bc.numpy().tolist()})
        config.update({'Wout': self.Wout.numpy().tolist()})
        config.update({'bout': self.bout.numpy().tolist()})
        
        return config


class SpectralGraphConv(layers.Layer):
    
    def __init__(self, input_dim, n_conv_feat, ord_conv, seed=None, name=None):
        
        super().__init__(trainable=True, name=name, dtype=DTYPE)
        
        self.input_dim = input_dim
        self.n_conv_feat = n_conv_feat
        self.ord_conv = ord_conv
        self.seed = seed
    
        self.list_lap = None
    
        self.init_trainable_weights(name)
        
    def init_trainable_weights(self, name):
      
        w_init = tf.keras.initializers.GlorotNormal(seed=self.seed)
        self.W = tf.Variable(initial_value=w_init(shape=(self.ord_conv * self.input_dim,
                                                         self.n_conv_feat), dtype=DTYPE),
                             name=f'{name}_W',
                             trainable=True)
                                                  
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(self.n_conv_feat,), dtype=DTYPE),
                             name=f'{name}_b',
                             trainable=True)
    
    def compute_cheb_polynomials(self, L):

        list_lap = []
        for k in range(self.ord_conv):
            
            if (k==0):
                list_lap.append(tf.cast(tf.linalg.tensor_diag(tf.ones([tf.shape(L)[0],])), DTYPE))
                
            elif (k==1):
                list_lap.append(tf.cast(L, DTYPE))
                
            else:
                list_lap.append(2 * tf.matmul(L, list_lap[k-1])  - list_lap[k-2])

        self.list_lap = list_lap
        assert self.list_lap is not None

    def call(self, inputs):
        """
        Args:
            A: self.W/self.H
        """
        
        if self.list_lap is None:
            raise ValueError('Should first compute polynomial terms!')
        
        feat = []
        # collect features
        for k in range(self.ord_conv):
            c_lap = self.list_lap[k]
            
            # dense implementation
            c_feat = tf.matmul(c_lap, inputs, a_is_sparse=False)
            feat.append(c_feat)
        
        all_feat = tf.concat(feat, 1)
        conv_feat = tf.matmul(all_feat, self.W) + self.b
        conv_feat = tf.nn.relu(conv_feat)

        return conv_feat

    def get_config(self):

        config = super(SpectralGraphConv, self).get_config()
        config.update({'W': self.W.numpy().tolist()})
        config.update({'b': self.b.numpy().tolist()})
        
        return config
