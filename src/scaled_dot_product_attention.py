"""
https://www.tensorflow.org/images/tutorials/transformer/scaled_attention.png"

The attention function used by the transformer takes three inputs: q (query), k (key), v (value). The equation used to
calculate the attention weights is:

    Attention(q, k, v) = softmax(q * k.T/ sqrt(d_k)) * V

The dot-product attention is scaled by a factor of square root of the depth. This is done because, for large values of
depth, the dot product grows large in magnitude and pushes the softmax function where it has small gradients, resulting
in a very hard softmax.

For example, consider that q and k have a mean of 0 and variance of 1: their matrix multiplication will have a mean of 0
and variance of d_k. The square root of d_k is then used for scaling, in order to get a consistent variance regardless
of the value of d_k: if the variance is too low, the output may be too flat to optimize effectively; on the other hand,
if the variance is too high, the softmax may saturate at initialization making it difficult to learn.

The mask is multiplied by -1e9 (close to negative infinity). This is done because the mask is summed with the scaled
matrix multiplication of q and k and is applied immediately before a softmax. The goal is to zero out these cells, and
large negative inputs to the softmax are near zero in the output.
"""
import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
  """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


if __name__ == '__main__':

    def print_out(q, k, v):
        temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
        print('Attention weights are:')
        print(temp_attn)
        print('Output is:')
        print(temp_out)

    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)
    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    # This query aligns with the second key, so the second value is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    print()

    # This query aligns with a repeated key (third and fourth), so all associated values get averaged.
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    print()

    # This query aligns equally with the first and second key, so their values get averaged.
    temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    print()

    temp_q = tf.constant([[0, 10, 0],
                          [0, 0, 10],
                          [10, 10, 0]], dtype=tf.float32)  # (3, 3)
    print_out(temp_q, temp_k, temp_v)
