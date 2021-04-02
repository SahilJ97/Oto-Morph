"""Written by Sahil Jayaram"""

import torch
from sigmorphon_seq2seq.joeynmt.decoders import MultiHeadRecurrentDecoder


class RNN(torch.nn.Module):
    def __init__(
            self,
            embed_size,
            hidden_size,
            n_chars,
            n_tags,
            init_lang_embeds,
            dropout=.3,
    ):
        super(RNN, self).__init__()
        self.character_encoder = torch.nn.LSTM(input_size=n_chars, hidden_size=embed_size / 2, batch_first=True,
                                               bidirectional=True)
        self.tagset_encoder = torch.nn.LSTM(input_size=n_tags, hidden_size=embed_size / 2, batch_first=True,
                                            bidirectional=True)
        self.decoder = MultiHeadRecurrentDecoder(rnn_type="lstm", emb_size=embed_size, hidden_size=hidden_size,
                                                 dropout=dropout, attention="luong", gate_func="entmax")
        self.lang_embeds = init_lang_embeds
        init_lang_embeds.requires_grad = True  # make language embeddings trainable

    def to(self, *args, **kwargs):
        self.character_encoder = self.character_encoder.to(*args, **kwargs)
        self.tagset_encoder = self.tagset_encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        self.lang_embeds = self.lang_embeds.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, inputs):
        lang_indices = inputs["language"]
        lang_embeds = torch.stack([self.lang_embeds[i] for i in lang_indices])
        char_seq = inputs["character_sequence"]
        tagset = inputs["tagset"]
        char_embeds = self.character_encoder(char_seq)  # (embeds, hidden) ?
        tagset_embeds = self.tagset_encoder(tagset)
