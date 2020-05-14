# encoding: utf-8
"""Class Declaration of Transformer's Decoder Block."""

import chainer

import chainer.functions as F

from espnet.nets.chainer_backend.transformer_wave.attention import MultiHeadAttention
from espnet.nets.chainer_backend.transformer_wave.layer_norm import LayerNorm
from espnet.nets.chainer_backend.transformer_wave.positionwise_feed_forward import PositionwiseFeedForward

import logging


class DecoderLayer(chainer.Chain):
    """Single decoder layer module.

    Args:
        n_units (int): Number of input/output dimension of a FeedForward layer.
        d_units (int): Number of units of hidden layer in a FeedForward layer.
        h (int): Number of attention heads.
        dropout (float): Dropout rate

    """

    def __init__(self, n_units, d_units=0, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        """Initialize DecoderLayer."""
        super(DecoderLayer, self).__init__()
        with self.init_scope():
            self.self_attn = MultiHeadAttention(n_units, h, dropout=dropout,
                                                initialW=initialW,
                                                initial_bias=initial_bias)
            self.src_attn = MultiHeadAttention(n_units, h, dropout=dropout,
                                               initialW=initialW,
                                               initial_bias=initial_bias)
            self.feed_forward = PositionwiseFeedForward(n_units, d_units=d_units,
                                                        dropout=dropout,
                                                        initialW=initialW,
                                                        initial_bias=initial_bias)
            self.norm1 = LayerNorm(n_units)
            self.norm2 = LayerNorm(n_units)
            self.norm3 = LayerNorm(n_units)
        self.dropout = dropout

    def forward(self, e, s, xy_mask, yy_mask, batch, cache=None):
        """Compute Encoder layer.

        Args:
            e (chainer.Variable): Batch of padded features. (B, Lmax)
            s (chainer.Variable): Batch of padded character. (B, Tmax)

        Returns:
            chainer.Variable: Computed variable of decoder.

        """
        n_e = self.norm1(e)
        if cache is None:
            n_e = self.self_attn(n_e, mask=yy_mask, batch=batch)
            e = e + F.dropout(n_e, self.dropout)
        else:
            # TODO(nelson): Implement batched forward
            assert batch == 1, f'Cached decoder is not implemented for {batch} samples'
            assert cache.shape == (
                n_e.shape[0] - 1,
                n_e.shape[1]
            ), f"{cache.shape} == {(n_e.shape[0] - 1, n_e.shape[1])}"
            ne_q = n_e[-1:]
            q_mask = None
            if yy_mask is not None:
                q_mask = yy_mask[:, -1:]
            n_e = self.self_attn(ne_q, s_var=n_e, mask=q_mask, batch=batch)
            e = e[-1:] + F.dropout(n_e, self.dropout)

        n_e = self.norm2(e)
        n_e = self.src_attn(n_e, s_var=s, mask=xy_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.norm3(e)
        n_e = self.feed_forward(n_e)
        e = e + F.dropout(n_e, self.dropout)
        
        if cache is not None:
            e = F.concat([cache, e], axis=0)
        return e
