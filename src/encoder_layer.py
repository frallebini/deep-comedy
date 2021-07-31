import tensorflow as tf

from multi_head_attention import MultiHeadAttention
from point_wise_ffn import point_wise_feed_forward_network


class EncoderLayer(tf.keras.layers.Layer):
    """
    Each encoder layer consists of:
        1. Multi-head attention (with padding mask).
        2. Point wise feed forward network.

    Each of these sublayers has a residual connection around it followed by a layer normalization. Residual connections
    help in avoiding the vanishing gradient problem in deep networks.

    There are ``num_layers`` encoder layers in the transformer.

    The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the ``d_model`` (last) axis.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


if __name__ == '__main__':
    sample_encoder_layer = EncoderLayer(512, 8, 2048)

    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)

    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
