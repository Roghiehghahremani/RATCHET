import argparse # Handles command-line arguments.
import imageio # Reads image files.
import glob # Helps find images in a directory.
import os
import datetime # Not used in the script (potential leftover).

import numpy as np
import tensorflow as tf # Loads models, processes images, and performs inference.

from model.transformer import Transformer, default_hparams # Loads a Transformer model for text generation.

from tokenizers import ByteLevelBPETokenizer # Loads a pre-trained tokenizer.

"""Loads a binary classifier (cxr_validator_model) that predicts whether an image is a chest X-ray (CXR).
Purpose: Ensures only CXR images are processed."""
def load_validator():
    validator_model = tf.keras.models.load_model('checkpoints/cxr_validator_model.tf')
    print('Validator Model Loaded!')
    return validator_model

"""oads the Byte-Pair Encoding (BPE) tokenizer for processing text.
Defines a Transformer model using hyperparameters (default_hparams).
Loads pre-trained weights from checkpoints/RATCHET2.tf."""
def load_model():

    # Load Tokenizer
    tokenizer = ByteLevelBPETokenizer(
        'preprocessing/mimic/mimic-vocab.json',
        'preprocessing/mimic/mimic-merges.txt',
    )

    # Load Model
    hparams = default_hparams()
    transformer = Transformer(
        num_layers=hparams['num_layers'],
        d_model=hparams['d_model'],
        num_heads=hparams['num_heads'],
        dff=hparams['dff'],
        target_vocab_size=tokenizer.get_vocab_size(),
        dropout_rate=hparams['dropout_rate'])
    transformer.load_weights('checkpoints/RATCHET2.tf')
    print(f'Model Loaded! Checkpoint file: checkpoints/RATCHET2.tf')

    return transformer, tokenizer

"""Filters logits so only the top-k highest probabilities remain.
Low-probability words are replaced with -1e10 (effectively ignored).
Helps prevent the model from picking low-confidence words."""
def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


"""Selects words whose cumulative probability is ≤ p.
Adaptive: Can include more words when the probability distribution is flat.
Ensures high-probability words are prioritized while allowing diversity."""
def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )

"""Starts with a <s> (start token).
Feeds input image and previous tokens to the Transformer to generate text word by word.
Applies top-k / top-p sampling and temperature scaling to adjust randomness.
Stops when the end token (2) is predicted.
Returns:
The decoded sentence (report).
The attention weights from the Transformer."""
def evaluate(inp_img, tokenizer, transformer, temperature, top_k, top_p, options, seed, MAX_LENGTH=128):

    # The first token to the transformer should be the start token
    output = tf.convert_to_tensor([[tokenizer.token_to_id('<s>')]])

    for _ in range(MAX_LENGTH):

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = transformer([inp_img, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1, :] / temperature  # (batch_size, vocab_size)
        predictions = top_k_logits(predictions, k=top_k)
        predictions = top_p_logits(predictions, p=top_p)

        if options == 'Greedy':
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)[:, tf.newaxis]
        elif options == 'Sampling':
            predicted_id = tf.random.categorical(predictions, num_samples=1, dtype=tf.int32, seed=seed)
        else:
            print('SHOULD NOT HAPPEN')

        # return the result if the predicted_id is equal to the end token
        if predicted_id == 2:  # stop token #tokenizer_en.vocab_size + 1:
            break

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    # transformer([inp_img, output[:, :-1]], training=False)
    return tf.squeeze(output, axis=0)[1:], transformer.decoder.last_attn_scores


"""Parses CLI arguments (temperature, top-k, top-p, etc.).
Loads the Transformer model & Validator.
Iterates through images in the input folder:
Skips non-image files.
Resizes & normalizes the image.
Validates whether it's a CXR (cxr_validator_model).
Generates a report using the Transformer.
Saves the report as a text file."""
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--options', default='Greedy')
    parser.add_argument('--inp_folder', default='inp_folder')
    parser.add_argument('--out_folder', default='out_folder')
    parser.add_argument('--temperature', default=1.)
    parser.add_argument('--top_k', default=0)
    parser.add_argument('--top_p', default=1.)
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()

    tf.config.set_visible_devices([], 'GPU')

    transformer, tokenizer = load_model()
    cxr_validator_model = load_validator()

    images = glob.glob(os.path.join(args.inp_folder, '*'))

    for image in images:

        if not image.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f'file {image} is not of image type: "png", "jpg" or "jpeg". Skipping... ')
            continue

        print(f'Generating Report for {os.path.basename(image)}')

        # Read input image with size [1, H, W, 1] and range (0, 255)
        img_array = imageio.imread(image, as_gray=True)[None, ..., None]

        # Convert image to float values in (0, 1)
        img_array = tf.image.convert_image_dtype(img_array.astype('uint8'), tf.float32)

        # Resize image with padding to [1, 224, 224, 1]
        img_array = tf.image.resize_with_pad(img_array, 224, 224, method=tf.image.ResizeMethod.BILINEAR)

        # Check image is CXR
        valid = tf.nn.sigmoid(cxr_validator_model(img_array))
        if valid < 0.1:
            continue

        # Generate radiology report
        result, attention_weights = evaluate(img_array, tokenizer, transformer,
                                             args.temperature, args.top_k, args.top_p,
                                             args.options, args.seed)
        predicted_sentence = tokenizer.decode(result)
        print(predicted_sentence)

        # Save report
        with open(os.path.join(args.out_folder, os.path.basename(image).split('.')[0] + '.txt'), 'w') as f:
            f.write(predicted_sentence)
