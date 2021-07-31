import tensorflow as tf

from decoder_layer import DecoderLayer
from encoder import Encoder
from positional_encoding import positional_encoding


class Decoder(tf.keras.layers.Layer):
    """
    A decoder consists of:
        1. Output Embedding.
        2. Positional Encoding.
        3. `num_layers` decoder layers.

    The target is put through an embedding which is summed with the positional encoding. The output of this summation is
    the input to the decoder layers. The output of the decoder is the input to the final linear layer.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


if __name__ == '__main__':
    sample_encoder = Encoder(num_layers=2,
                             d_model=512,
                             num_heads=8,
                             dff=2048,
                             input_vocab_size=8500,
                             maximum_position_encoding=10000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    sample_decoder = Decoder(num_layers=2,
                             d_model=512,
                             num_heads=8,
                             dff=2048,
                             target_vocab_size=8000,
                             maximum_position_encoding=5000)

    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                  enc_output=sample_encoder_output,
                                  training=False,
                                  look_ahead_mask=None,
                                  padding_mask=None)

    print(output.shape, attn['decoder_layer2_block2'].shape)
