"""Written by Sahil Jayaram"""

import torch
from entmax import sparsemax


class Decoder(torch.nn.Module):
    def __init__(self, embed_size, n_chars, dropout=None, beam_size=10):
        super().__init__()
        self.n_chars = n_chars
        self.beam_size = beam_size
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
        current_input = torch.zeros((batch_size, self.n_chars), device=char_encoding.device)
        last_cell_state = (
            #torch.cat((char_hn[0], tag_hn[0]), dim=-1),
            #torch.cat((char_cn[0], tag_cn[0]), dim=-1)
            torch.cat((char_hn[0], char_hn[1]), dim=-1),
            torch.cat((char_cn[0], char_cn[1]), dim=-1)
        )

        def time_step_fn(input_1, state_0):
            h1, c1 = self.lstm_cell(input_1, state_0)
            query = torch.unsqueeze(h1, dim=0)  # use cell output as query
            char_attention, _ = self.char_attention(query=query, key=char_encoding, value=char_encoding)
            tag_attention, _ = self.tag_attention(query=query, key=tag_encoding, value=tag_encoding)
            aggregated_attention = torch.cat([char_attention, tag_attention], dim=-1).squeeze(0)
            #aggregated_attention = sparsemax(aggregated_attention, dim=-1)
            aggregated_attention = torch.relu(aggregated_attention)
            output = self.output_layer(aggregated_attention)  # relu instead?
            return output, (h1, c1)

        top = [[current_input, last_cell_state, [], 0]]  # beam search candidates; last entry is log probability
        teacher_forcing = true_output_seq is not None
        for time_step in range(len(char_encoding)):
            time_step_leaders = []
            for candidate in top:
                next_input, current_cell_state, current_output_seq, sequence_probability = candidate
                candidate_output, candidate_next_state = time_step_fn(next_input, current_cell_state)
                if teacher_forcing:  # teacher forcing; in this scenario, top only has 1 item
                    top = [[None, candidate_next_state, current_output_seq + [candidate_output], 1]]
                    if time_step < len(char_encoding) - 1:
                        top[0][0] = true_output_seq[:, time_step + 1, :]
                    continue
                else:
                    top_vals, top_indices = torch.topk(candidate_output, self.beam_size, dim=-1)
                    for i in range(self.beam_size):
                        time_step_leaders.append(
                            [top_indices[0][i], top_vals[0][i], candidate_next_state, current_output_seq,
                             sequence_probability + torch.log(top_vals[0][i])]
                        )
            if not teacher_forcing:
                new_top = []
                time_step_leaders.sort(key=lambda x: x[4])
                beam_size = self.beam_size
                if time_step == self.beam_size - 1:
                    beam_size = 1
                for leader in time_step_leaders[-beam_size:]:
                    leader_index, leader_prob, leader_next_state, leader_current_output_seq, probability = leader
                    one_hot = torch.nn.functional.one_hot(leader_index, self.n_chars)
                    one_hot = torch.unsqueeze(one_hot, dim=0).float()
                    new_top.append([one_hot, leader_next_state, leader_current_output_seq + [one_hot], probability])
                top = new_top

        return_sequence = top[0][2]
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
            beam_size=10
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
        self.decoder = Decoder(embed_size, n_chars, dropout=dropout, beam_size=beam_size)
        if type(init_lang_embeds) == list:
            init_lang_embeds = torch.stack(init_lang_embeds)
        self.register_parameter(name="lang_embeds", param=torch.nn.Parameter(init_lang_embeds))

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
        batch_size = len(char_seq)
        enhanced_char_seq, enhanced_tag_seq = [], []
        selected_lang_embeds = torch.index_select(self.lang_embeds, dim=0, index=lang_indices)
        for i in range(batch_size):
            lang_embed = selected_lang_embeds[i]
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
