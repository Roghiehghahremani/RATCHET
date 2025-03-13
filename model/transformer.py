from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime # Used to handle date and time operations (e.g., logging).

import numpy as np
import tensorflow as tf

"""This function defines the default hyperparameters for the transformer model. The keys indicate image dimensions, model parameters, and architecture details:
img_x, img_y: Image dimensions (224x224).
img_ch: Number of image channels (1, for grayscale).
d_model: Dimensionality of the model (embedding size).
dff: Dimensionality of the feed-forward layer.
num_heads: Number of attention heads in multi-head attention.
num_layers: Number of layers in the decoder.
dropout_rate: Dropout rate used to prevent overfitting."""
def default_hparams():
    return {
        'img_x': 224,
        'img_y': 224,
        'img_ch': 1,
        'd_model': 512,
        'dff': 2048,
        'num_heads': 8,
        'num_layers': 6,
        'dropout_rate': 0.1
    }

"""Positional encoding adds information about the relative position of tokens (e.g., in text or spatial locations for images) to the input embeddings. This is essential for transformers since they don't inherently have any notion of the position of tokens. The function:
Generates sine and cosine functions at different frequencies to represent positional encodings.
Concatenates sine and cosine encodings and returns the result."""
def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

"""This custom Keras layer adds positional encodings to token embeddings. It:
Initializes a word embedding layer and adds positional encoding.
Scales the embedding values by the square root of the embedding dimension to control the scale of the input to the network.
Adds positional encoding to the embeddings to represent the order of the tokens."""
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

"""This class serves as the base for both types of attention layers. It initializes:
MultiHeadAttention: Implements the multi-head attention mechanism.
LayerNormalization: Normalizes the output of the attention mechanism.
Add: Adds the input x to the output of the attention mechanism (residual connection)."""
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

"""This performs "cross-attention," where:
x is the query, which is updated using context (keys and values).
The attention scores are cached for later visualization or analysis.
The residual connection and layer normalization are applied."""
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

"""Causal self-attention is used to ensure that the model only attends to previous tokens in the sequence 
(i.e., it does not look ahead), which is critical for tasks like language modeling."""
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

"""This class implements the feed-forward layers that are used after attention layers. It consists of:
A fully connected layer with ReLU activation.
A second fully connected layer that reduces the dimensionality back to d_model.
Dropout and layer normalization are applied to prevent overfitting and stabilize training."""
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

"""This class represents a single layer of the decoder, containing:
Causal self-attention.
Cross-attention.
Feed-forward network.
The call method performs these operations sequentially and applies residual connections and layer normalization."""
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x

"""The encoder takes an image as input, uses a DenseNet-121 model for feature extraction, and applies a fully connected layer 
to the extracted features. If pre-trained weights are provided, they are loaded into the DenseNet-121 model."""
class Encoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, input_shape, pretrain_weights=None):
        super(Encoder, self).__init__()

        # shape after fc == (batch_size, nf * nf, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')

        # Use DenseNet-121 as feature extraction model
        self.base_model = tf.keras.applications.DenseNet121(
            include_top=False, weights=None, input_shape=input_shape)

        # Load pre-trained weights if present
        if pretrain_weights:
            print(f'{datetime.datetime.now()}: I Loading Pretrained DenseNet-121 weights: {pretrain_weights}')
            self.base_model.load_weights(pretrain_weights)
        else:
            print(f'{datetime.datetime.now()}: I No Pretrained DenseNet-121 weights specified')

    def call(self, x, **kwargs):
        x = self.base_model(x)
        # DenseNet-121 output is (batch_size, ?, ?, 1024)
        s = tf.shape(x)
        x = tf.reshape(x, (s[0], s[1] * s[2], x.shape[3]))
        x = self.fc(x)
        return x

"""The decoder generates a sequence output based on the context from the encoder. It uses positional embeddings and 
applies multiple layers of attention and feed-forward networks."""
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x

"""This is the main transformer model. It:

Uses the Encoder to process the input image (or context).
Uses the Decoder to generate the output sequence (possibly text or other representations).
The final output is processed through a fully connected layer to get logits (predictions)."""
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, dropout_rate=0.1, input_shape=(224, 224, 1),
                 classifier_weights=None):
        super(Transformer, self).__init__()

        self.encoder = Encoder(d_model, input_shape,
                               pretrain_weights=classifier_weights)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits

"""This is the main transformer model. It:

Uses the Encoder to process the input image (or context).
Uses the Decoder to generate the output sequence (possibly text or other representations).
The final output is processed through a fully connected layer to get logits (predictions)."""
if __name__ == "__main__":

    hparams = default_hparams()

    transformer = Transformer(
        num_layers=hparams['num_layers'],
        d_model=hparams['d_model'],
        num_heads=hparams['num_heads'],
        dff=hparams['dff'],
        target_vocab_size=2048,
        dropout_rate=hparams['dropout_rate'])

    a=1


    image = np.random.rand(1,224,224,1).astype('float32')
    text = np.random.randint(0, 2048, size=(1, 27))

    output = transformer((image, text))
