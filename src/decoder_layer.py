import tensorflow as tf

from encoder_layer import EncoderLayer
from multi_head_attention import MultiHeadAttention
from point_wise_ffn import point_wise_feed_forward_network


class DecoderLayer(tf.keras.layers.Layer):
    """
    Each decoder layer consists of:
        1. Masked multi-head attention (with look ahead mask and padding mask).
        2. Multi-head attention (with padding mask). v (value) and k (key) receive the encoder output as inputs.
           q (query) receives the output from the masked multi-head attention sublayer.
        3. Point wise feed forward network.

    Each of these sublayers has a residual connection around it followed by a layer normalization. The output of each
    sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the ``d_model`` (last) axis.

    There are ``num_layers`` decoder layers in the transformer.

    As q receives the output from decoder's first attention block, and k receives the encoder output, the attention
    weights represent the importance given to the decoder's input based on the encoder's output. In other words, the
    decoder predicts the next word by looking at the encoder output and self-attending to its own input.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


if __name__ == '__main__':
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)

    sample_decoder_layer = DecoderLayer(512, 8, 2048)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(tf.random.uniform((64, 50, 512)),
                                                             sample_encoder_layer_output,
                                                             False, None, None)

    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)
