import tensorflow as tf

from encoder_layer import EncoderLayer
from positional_encoding import positional_encoding


class Encoder(tf.keras.layers.Layer):
    """
    An encoder consists of:
        1. Input Embedding.
        2. Positional Encoding.
        3. `num_layers` encoder layers.

    The input is put through an embedding which is summed with the positional encoding. The output of this summation is
    the input to the encoder layers. The output of the encoder is the input to the decoder.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


if __name__ == '__main__':
    sample_encoder = Encoder(num_layers=2,
                             d_model=512,
                             num_heads=8,
                             dff=2048,
                             input_vocab_size=8500,
                             maximum_position_encoding=10000)

    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
