"""
Since this model doesn't contain any recurrence or convolution, a positional encoding is added to give the model some
information about the relative position of the words in the sentence.

The positional encoding vector is added to the embedding vector. Embeddings represent a token in a d-dimensional space,
where tokens with similar meaning are closer to each other; however, the embeddings do not encode the relative position
of words in a sentence. Therefore, after adding the positional encoding, words will be closer to each other — in the
d-dimensional space — based on both the similarity of their meaning and their position in the sentence.

The formula for calculating the positional encoding is as follows:

    PE(pos, 2i) = sin(pos / 10000^(2i / d_model)
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model)

where pos is the position and i is the dimension.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array: 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array: 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


if __name__ == '__main__':
    n, d = 2048, 512
    pos_encoding = positional_encoding(n, d)
    print(pos_encoding.shape)
    pos_encoding = pos_encoding[0]

    # Juggle the dimensions for the plot
    pos_encoding = tf.reshape(pos_encoding, (n, d // 2, 2))
    pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
    pos_encoding = tf.reshape(pos_encoding, (d, n))

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()
