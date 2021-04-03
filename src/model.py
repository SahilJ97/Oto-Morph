"""Written by Sahil Jayaram"""

import torch
from entmax import entmax15


class Decoder(torch.nn.Module):
    def __init__(self, embed_size, n_chars, dropout=None):
        super().__init__()
        self.n_chars = n_chars
        self.lstm_cell = torch.nn.LSTMCell(n_chars, embed_size)
        self.char_attention = torch.nn.MultiheadAttention(embed_dim=embed_size, num_heads=1, dropout=dropout)
        self.tag_attention = torch.nn.MultiheadAttention(embed_dim=embed_size, num_heads=1, dropout=dropout)
        self.output_layer = torch.nn.Linear(embed_size*2, n_chars)

    def to(self, *args, **kwargs):
        self.lstm_cell = self.lstm_cell.to(*args, **kwargs)
        self.char_attention = self.char_attention.to(*args, **kwargs)
        self.tag_attention = self.tag_attention.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, char_encoder_result, tag_encoder_result, true_output_seq=None):
        char_encoding, (char_hn, char_cn) = char_encoder_result
        batch_size = len(char_encoding)
        tag_encoding, (tag_hn, tag_cn) = tag_encoder_result
        char_encoding = torch.transpose(char_encoding, 0, 1)  # move seq_len dimension to the front
        tag_encoding = torch.transpose(tag_encoding, 0, 1)
        return_sequence = []
        current_input = torch.zeros((batch_size, self.n_chars))
        last_cell_state = (
            torch.cat((char_hn[0], tag_hn[0]), dim=-1),
            torch.cat((char_cn[0], tag_cn[0]), dim=-1)
        )
        for time_step in range(len(char_encoding)):  # decoder output sequence should be as long as character encoding
            h1, c1 = self.lstm_cell(current_input, last_cell_state)
            last_cell_state = (h1, c1)
            cell_output = torch.unsqueeze(h1, dim=0)
            char_attention, _ = self.char_attention(query=cell_output, key=char_encoding, value=char_encoding)
            tag_attention, _ = self.tag_attention(query=cell_output, key=tag_encoding, value=tag_encoding)
            aggregated_attention = torch.cat([char_attention, tag_attention], dim=-1).squeeze(0)
            output = self.output_layer(aggregated_attention)
            output = entmax15(output, dim=-1)
            return_sequence.append(output)
            if true_output_seq is None:
                current_input = output
            elif time_step < len(char_encoding) - 1:  # teacher forcing
                current_input = true_output_seq[:, time_step + 1, :]
        return_sequence = torch.stack(return_sequence)
        return torch.transpose(return_sequence, 0, 1)


class RNN(torch.nn.Module):
    def __init__(
            self,
            embed_size,
            n_chars,
            n_tags,
            init_lang_embeds,
            dropout=.3,
    ):
        super(RNN, self).__init__()
        self.lang_embeds = init_lang_embeds
        lang_dim = len(init_lang_embeds[0])
        if embed_size % 2 != 0:
            print("Embedding size must be even!")
            exit(1)
        self.character_encoder = torch.nn.LSTM(input_size=n_chars+lang_dim, hidden_size=embed_size//2, batch_first=True,
                                               bidirectional=True)
        # originally bidirectional. consider reverting. not hard! double hidden size of decoder,
        # use states from both encoders.
        self.tagset_encoder = torch.nn.LSTM(input_size=n_tags+lang_dim, hidden_size=embed_size//2, batch_first=True,
                                            bidirectional=True)
        self.decoder = Decoder(embed_size, n_chars, dropout=dropout)
        init_lang_embeds.requires_grad = True  # make language embeddings trainable

    def to(self, *args, **kwargs):
        self.character_encoder = self.character_encoder.to(*args, **kwargs)
        self.tagset_encoder = self.tagset_encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        self.lang_embeds = self.lang_embeds.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, inputs, labels=None):
        # Concatenate language embeddings to each vector
        lang_indices = inputs["language"]
        char_seq = inputs["character_sequence"]
        tagset = inputs["tagset"]
        char_seq_len = len(char_seq[0])
        tag_seq_len = len(tagset[0])
        lang_embeds = torch.stack([self.lang_embeds[i] for i in lang_indices])
        lang_embeds_for_char = torch.stack(
            [lang_embed.repeat(char_seq_len, 1) for lang_embed in lang_embeds]
        )
        lang_embeds_for_tag = torch.stack(
            [lang_embed.repeat(tag_seq_len, 1) for lang_embed in lang_embeds]
        )
        char_seq = torch.cat([char_seq, lang_embeds_for_char], dim=-1)
        tagset = torch.cat([tagset, lang_embeds_for_tag], dim=-1)

        # Encode character sequence and tagset
        char_encoder_result = self.character_encoder(char_seq)
        tag_encoder_result = self.tagset_encoder(tagset)

        # Decode
        return self.decoder(char_encoder_result, tag_encoder_result, true_output_seq=labels)


if __name__ == "__main__":
    lang_embeds = torch.tensor([[1., 2., 3.]])  # one language (3d embedding)
    model = RNN(embed_size=6, n_chars=2, n_tags=2, init_lang_embeds=lang_embeds)
    input_dict = {
        "language": [0, 0],  # one language index for each item in batch
        "character_sequence": torch.tensor([[[0, 1], [1, 0], [1, 0]], [[0, 1], [1, 0], [1, 0]]], dtype=torch.float),
        "tagset": torch.tensor([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], dtype=torch.float)
    }
    print(model.forward(input_dict))
