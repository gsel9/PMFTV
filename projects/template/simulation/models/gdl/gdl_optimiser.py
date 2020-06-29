import tensorflow as tf


def get_gdl_optimiser(optimiser, learning_rate):

    if optimiser == 'Adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    raise ValueError(f"Did not recognise optimiser: {optimiser}")