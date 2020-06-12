from abc import ABC, abstractmethod

import numpy as np

from .special_matrices import finite_difference_matrix, laplacian_kernel_matrix


SEED = 0
tf.random.set_seed(SEED)

DTYPE = tf.float32


class BaseLoss(ABC):

    def __init__(self, name="loss_function"):

        self.name = name

    def __call__(self, prediction, target, **kwargs):
        return self._build_loss(prediction, target, **kwargs)

    @abstractmethod
    def _build_loss(self, prediction, target):
        pass


class OriginalLoss(BaseLoss):

    def __init__(self, O, Lr, Lc, col_gamma, row_gamma, name="OriginalLoss",**kwargs):

        super().__init__(name=name)

        self.O = O
        self.Lr = Lr 
        self.Lc = Lc
        self.col_gamma
        self.row_gamma

    def _build_loss(self, Y_true, Y_pred, *args):
        
        frob_tensor = tf.multiply(self.O, Y_true - Y_pred)
        loss_frob = tf.square(frobenius_norm(frob_tensor)) / np.sum(self.O)
        
        trace_col_tensor = tf.matmul(tf.matmul(Y_pred, self.Lc), Y_pred, transpose_b=True)
        loss_trace_col = self.col_gamma * tf.linalg.trace(trace_col_tensor) / tf.size(Y_pred, out_type=DTYPE)
        
        trace_row_tensor = tf.matmul(tf.matmul(Y_pred, self.Lr, transpose_a=True), Y_pred)
        loss_trace_row = self.row_gamma * tf.linalg.trace(trace_row_tensor) / tf.size(Y_pred, out_type=DTYPE)

        return loss_frob + loss_trace_row + loss_trace_col


class TVLoss(BaseLoss):

    def __init__(self, O, Lr, Lc, col_gamma, row_gamma, epsilon=1e-9, name="TVLoss", **kwargs):

        super().__init__(name=name)

        self.O = O
        self.Lr = Lr 
        self.Lc = Lc
        self.col_gamma
        self.row_gamma
        self.tv_gamma = tv_gamma
        self.epsilon = epsilon

    def _build_loss(self, Y_true, Y_pred, V, *args):
        
        frob_tensor = tf.multiply(self.O, Y_true - self.Y_pred)
        loss_frob = tf.square(frobenius_norm(frob_tensor)) / np.sum(self.O)
        
        trace_col_tensor = tf.matmul(tf.matmul(Y_pred, self.Lc), Y_pred, transpose_b=True)
        loss_trace_col = self.col_gamma * tf.linalg.trace(trace_col_tensor) / tf.size(Y_pred, out_type=DTYPE)
        
        trace_row_tensor = tf.matmul(tf.matmul(Y_pred, self.Lr, transpose_a=True), Y_pred)
        loss_trace_row = self.row_gamma * tf.linalg.trace(trace_row_tensor) / tf.size(Y_pred, out_type=DTYPE)
        
        col_diff_tensor = tf.sqrt(np.diff(V, axis=1) ** 2 + self.epsilon ** 2)
        loss_tv = self.tv_gamma * tf.reduce_sum(col_diff_tensor) / tf.size(V, out_type=DTYPE)

        return loss_frob + loss_trace_row + loss_trace_col + loss_tv 


class ConvolutionLoss(BaseLoss):

    def __init__(self, O, Lr, Lc, col_gamma, row_gamma, name="ConvolutionLoss", **kwargs):

        super().__init__(name=name)

        self.O = O
        self.Lr = Lr 
        self.Lc = Lc
        self.col_gamma
        self.row_gamma
        self.conv_gamma = conv_gamma
 
        C = laplacian_kernel_matrix(self.Y_true.shape[1])
        R = finite_difference_matrix(self.Y_true.shape[1])
        self.CR = C @ R

    def _build_loss(self, Y_true, Y_pred, V, *args):
        
        frob_tensor = tf.multiply(self.O, Y_true - self.Y_pred)
        loss_frob = tf.square(frobenius_norm(frob_tensor)) / np.sum(self.O)
        
        trace_col_tensor = tf.matmul(tf.matmul(Y_pred, self.Lc), Y_pred, transpose_b=True)
        loss_trace_col = self.col_gamma * tf.linalg.trace(trace_col_tensor) / tf.size(Y_pred, out_type=DTYPE)
        
        trace_row_tensor = tf.matmul(tf.matmul(Y_pred, self.Lr, transpose_a=True), Y_pred)
        loss_trace_row = self.row_gamma * tf.linalg.trace(trace_row_tensor) / tf.size(Y_pred, out_type=DTYPE)

        loss_conv = self.conv_gamma * tf.square(frobenius_norm(self.CR @ V)) / tf.size(V, out_type=DTYPE)

        return loss_frob + loss_trace_row + loss_trace_col + loss_conv 


class TVConvolutionLoss(BaseLoss):

    def __init__(self, O, Lr, Lc, tv_gamma, col_gamma, row_gamma, epsilon=1e-9, name="TVConvolutionLoss", **kwargs):

        super().__init__(name=name)

        self.O = O
        self.Lr = Lr 
        self.Lc = Lc
        self.col_gamma
        self.row_gamma
        self.tv_gamma = tv_gamma
        self.conv_gamma = conv_gamma
        self.epsilon = epsilon

        C = laplacian_kernel_matrix(self.Y_true.shape[1])
        R = finite_difference_matrix(self.Y_true.shape[1])
        self.CR = C @ R

    def _build_loss(self, Y_true, Y_pred, V, *args):
        
        frob_tensor = tf.multiply(self.O, Y_true - self.Y_pred)
        loss_frob = tf.square(frobenius_norm(frob_tensor)) / np.sum(self.O)
        
        trace_col_tensor = tf.matmul(tf.matmul(Y_pred, self.Lc), Y_pred, transpose_b=True)
        loss_trace_col = self.col_gamma * tf.linalg.trace(trace_col_tensor) / tf.size(Y_pred, out_type=DTYPE)
        
        trace_row_tensor = tf.matmul(tf.matmul(Y_pred, self.Lr, transpose_a=True), Y_pred)
        loss_trace_row = self.row_gamma * tf.linalg.trace(trace_row_tensor) / tf.size(Y_pred, out_type=DTYPE)
        
        loss_conv = self.conv_gamma * tf.square(frobenius_norm(self.CR @ V)) / tf.size(V, out_type=DTYPE)

        col_diff_tensor = tf.sqrt(np.diff(V, axis=1) ** 2 + self.epsilon ** 2)
        loss_tv = self.tv_gamma * tf.reduce_sum(col_diff_tensor) / tf.size(V, out_type=DTYPE)

        return loss_frob + loss_trace_row + loss_trace_col + loss_conv + loss_tv
