import numpy as np
import tensorflow as tf

def create_angles_rates(d_model):
    angles = np.arange(d_model)
    angles = 1 / (10000 ** (angles / d_model))
    angles = np.expand_dims(angles, axis=0)
    return angles

def generate_positional_encoding(q_length, d_model):
    """
        Parameters
        ----------
        q_length : int
            - The length of an input
        d_model : int
            - Embedding dimension
        Returns
        ----------
        pos_angles: tensor
            - positional embedding matrix
    """
    angles = create_angles_rates(d_model)
    pos = np.expand_dims(np.arange(q_length), axis=1)
    pos_angles = pos.dot(angles)
    pos_angles[:, 0::2] = np.sin(pos_angles[:, 0::2])
    pos_angles[:, 1::2] = np.cos(pos_angles[:, 1::2])
    pos_angles = np.expand_dims(pos_angles, axis=1)
    return tf.cast(pos_angles, dtype=tf.float32)
