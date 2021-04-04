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
        self.output_layer = self.output_layer.to(*args, **kwargs)
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
        )  # use final states for left-to-right direction
        for time_step in range(len(char_encoding)):  # decoder output sequence should be as long as input sequence
            h1, c1 = self.lstm_cell(current_input, last_cell_state)
            last_cell_state = (h1, c1)
            query = torch.unsqueeze(h1, dim=0)  # use cell output as query
            char_attention, _ = self.char_attention(query=query, key=char_encoding, value=char_encoding)
            tag_attention, _ = self.tag_attention(query=query, key=tag_encoding, value=tag_encoding)
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
            init_lang_embeds,  # list of tensors
            dropout=.3,
    ):
        super(RNN, self).__init__()
        lang_dim = len(init_lang_embeds[0])
        if embed_size % 2 != 0:
            print("Embedding size must be even!")
            exit(1)
        self.character_encoder = torch.nn.LSTM(input_size=n_chars+lang_dim, hidden_size=embed_size//2, batch_first=True,
                                               bidirectional=True)
        self.tagset_encoder = torch.nn.LSTM(input_size=n_tags+lang_dim, hidden_size=embed_size//2, batch_first=True,
                                            bidirectional=True)
        self.decoder = Decoder(embed_size, n_chars, dropout=dropout)
        if type(init_lang_embeds) == list:
            init_lang_embeds = torch.stack(init_lang_embeds)
        self.register_parameter(name="lang_embeds", param=torch.nn.Parameter(init_lang_embeds))

    def to(self, *args, **kwargs):
        self.character_encoder = self.character_encoder.to(*args, **kwargs)
        self.tagset_encoder = self.tagset_encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        for i in range(len(self.lang_embeds)):
            self.lang_embeds[i] = self.lang_embeds[i].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, inputs, labels=None):
        # Concatenate language embeddings to each vector
        lang_indices = inputs["language"]
        char_seq = inputs["character_sequence"]
        tagset = inputs["tagset"]
        char_seq_len = len(char_seq[0])
        tag_seq_len = len(tagset[0])
        batch_size = len(char_seq)
        enhanced_char_seq, enhanced_tag_seq = [], []
        selected_lang_embeds = torch.index_select(self.lang_embeds, dim=0, index=lang_indices)
        for i in range(batch_size):
            lang_embed = selected_lang_embeds[lang_indices[i]]
            new_char, new_tag = [], []
            for j in range(char_seq_len):
                new_char.append(torch.cat([char_seq[i][j], lang_embed], dim=-1))
            for j in range(tag_seq_len):
                new_tag.append(torch.cat([tagset[i][j], lang_embed], dim=-1))
            enhanced_char_seq.append(torch.stack(new_char))
            enhanced_tag_seq.append(torch.stack(new_tag))
        char_seq = torch.stack(enhanced_char_seq)
        tagset = torch.stack(enhanced_tag_seq)

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
