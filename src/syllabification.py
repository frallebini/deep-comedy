"""
Just some functions from ``syllabification.ipynb`` to allow to call the syllabifier in ``generation.ipynb``.
"""
import tensorflow as tf
import re
from masking import create_masks


def syllabify(verse, syllabifier, tokenizer_nosyll, tokenizer_syll):
    encoder_input = tokenizer_nosyll.texts_to_sequences([verse])
    encoder_input = tf.convert_to_tensor(encoder_input)

    start = tf.constant(tokenizer_syll.word_index['<'], dtype=tf.int64)
    end = tf.constant(tokenizer_syll.word_index['>'], dtype=tf.int64)

    output = tf.convert_to_tensor([start], dtype=tf.int64)
    output = tf.expand_dims(output, axis=0)

    predicted_id = 0
    while predicted_id != end:  # return the result if the predicted_id is equal to the end-of-verse token
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, _ = syllabifier(encoder_input,
                                     output,
                                     False,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)

        # select the last character from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.argmax(predictions, axis=-1)

        # concatenate the predicted_id to the output which is given to the decoder as its input
        output = tf.concat([output, predicted_id], axis=-1)

    # output.shape (1, tokens)
    syllabified_verse = sequences_to_text(tokenizer_syll, output.numpy())

    return syllabified_verse


def cleanup(verse):
    # remove verse numeration and add start-of-verse ('<') and end-of-verse ('>') tokens
    return f"<{re.sub(r'[0-9]+', '', verse).strip()}>"


def collect_verses(path):
    with open(path) as f:
        # remove blank lines and headers
        return [cleanup(line) for line in f if line != '\n' and '•' not in line]


def tokenize(verses):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False, char_level=True)
    tokenizer.fit_on_texts(verses)
    return tokenizer


def get_tokenizers():
    url = 'https://raw.githubusercontent.com/asperti/Dante/main'

    names_nosyll = ['inferno.txt', 'purgatorio.txt', 'paradiso.txt']
    names_syll = ['inferno_syllnew.txt', 'purgatorio_syllnew.txt', 'paradiso_syllnew.txt']

    paths_nosyll = [tf.keras.utils.get_file(name, origin=f'{url}/{name}') for name in names_nosyll]
    paths_syll = [tf.keras.utils.get_file(name, origin=f'{url}/{name}') for name in names_syll]

    verses_nosyll = [verse for path in paths_nosyll for verse in collect_verses(path)]
    verses_syll = [verse for path in paths_syll for verse in collect_verses(path)]
    
    tokenizer_nosyll = tokenize(verses_nosyll)
    tokenizer_syll = tokenize(verses_syll)

    return tokenizer_nosyll, tokenizer_syll


def sequences_to_text(tokenizer, sequences):
    return [s.replace('  ', '*').replace(' ', '').replace('*', ' ') for s in tokenizer.sequences_to_texts(sequences)]
