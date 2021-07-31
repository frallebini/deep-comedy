"""
https://www.tensorflow.org/images/tutorials/transformer/transformer.png
"""
import tensorflow as tf

from decoder import Decoder
from encoder import Encoder


class Transformer(tf.keras.Model):
    """
    A transformer consists of:
        1. Encoder.
        2. Decoder.
        3. Final linear layer.

    The output of the decoder is the input to the linear layer and its output is returned.
    """
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size, target_vocab_size,
                 pe_input, pe_target,
                 rate=0.1):

        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               input_vocab_size,
                               pe_input,
                               rate)
        self.decoder = Decoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               target_vocab_size,
                               pe_target,
                               rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


if __name__ == '__main__':
    sample_transformer = Transformer(num_layers=2,
                                     d_model=512,
                                     num_heads=8,
                                     dff=2048,
                                     input_vocab_size=8500, target_vocab_size=8000,
                                     pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer(temp_input, temp_target,
                                   training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
