"""Emotion-conditioned motion network definitions."""

import torch
import torch.nn as nn
import math


def init_biased_mask(n_head, max_seq_len, T):
    """Build the temporally biased attention mask used by the motion decoder."""

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=T).unsqueeze(1).repeat(
        1, T
    ).view(-1) // (T)
    bias = -torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, : i + 1] = bias[-(i + 1) :]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    mask = mask.unsqueeze(0) + alibi
    return mask


def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, i] = 0
    return (mask == 1).to(device=device)


class TicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, T=25, max_seq_len=600):
        super(TicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(T, d_model)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, d_model)
        repeat_num = (max_seq_len // T) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class EmotionMotionPredictor(nn.Module):
    def __init__(self, args):
        super(EmotionMotionPredictor, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.n_head = 4
        self.audio_feature_map = nn.Linear(512, args.feature_dim)
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.PPE = TicPositionalEncoding(args.feature_dim, T=args.T)
        self.biased_mask = init_biased_mask(
            n_head=self.n_head, max_seq_len=600, T=args.T
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.feature_dim,
            nhead=self.n_head,
            dim_feedforward=2 * args.feature_dim,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        self.emo_type = args.emo_type
        self.emo_num = args.emo_num
        if self.emo_type == 1:
            self.emo_vector = nn.Linear(args.emo_num, args.feature_dim)
        elif self.emo_type == 2:
            self.audio_former = AudioConditionEncoder(args)
        elif self.emo_type == 3:
            self.audio_emo_former = EmotionAudioConditionEncoder(args)
        if args.style_template:
            args.style_dim = args.feature_dim
            args.style_T = 100
            self.obj_vector = SpeakingStyleEncoder(args)
        else:
            self.obj_vector = nn.Linear(
                args.person_num,
                args.feature_dim,
                bias=getattr(args, "person_embedding_bias", True),
            )
        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)
        self.style_template = args.style_template

    def forward(self, audio, template, vertice, one_hot, emo, teacher_forcing=False):
        vertice = vertice.reshape(vertice.shape[0], vertice.shape[1], -1)
        template = template[:, 0].reshape(template.shape[0], -1)
        if self.style_template:
            obj_embedding = self.obj_vector(one_hot)  # (1, feature_dim)
        else:
            obj_embedding = self.obj_vector(one_hot)  # (1, feature_dim)
        if self.emo_type == 1:
            emo_embedding = self.emo_vector(emo)
            obj_embedding = obj_embedding + emo_embedding
        frame_num = vertice.shape[1]

        if self.emo_type == 3:
            hidden_states = self.audio_emo_former(audio, emo, teacher_forcing)
        elif self.emo_type == 2:
            hidden_states = self.audio_former(audio, emo)
        else:
            hidden_states = self.audio_feature_map(audio)

        if teacher_forcing:
            vertice_emb = obj_embedding.unsqueeze(1)  # (1,1,feature_dim)
            style_emb = vertice_emb
            vertice_input = torch.cat(
                (template, vertice[:, :-1]), 1
            )  # shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            tgt_mask = (
                self.biased_mask[:, : vertice_input.shape[1], : vertice_input.shape[1]]
                .clone()
                .detach()
                .to(device=self.device)
            )
            memory_mask = enc_dec_mask(
                self.device, vertice_input.shape[1], hidden_states.shape[1]
            )

            vertice_out = self.transformer_decoder(
                vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask
            )
            vertice_out = self.vertice_map_r(vertice_out)
        else:
            for i in range(frame_num):
                if i == 0:
                    vertice_emb = obj_embedding.unsqueeze(1)  # (1,1,feature_dim)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)
                tgt_mask = (
                    self.biased_mask[
                        :, : vertice_input.shape[1], : vertice_input.shape[1]
                    ]
                    .clone()
                    .detach()
                    .to(device=self.device)
                )
                tgt_mask = tgt_mask.repeat(vertice_input.shape[0], 1, 1)
                memory_mask = enc_dec_mask(
                    self.device, vertice_input.shape[1], hidden_states.shape[1]
                )
                vertice_out = self.transformer_decoder(
                    vertice_input,
                    hidden_states,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                )
                vertice_out = self.vertice_map_r(vertice_out)
                new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb

                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out.reshape(
            vertice_out.shape[0], vertice_out.shape[1], 3, -1
        )

        return vertice_out

    def predict(self, audio, template, one_hot, emo, vertice_init=None):
        template = template[:, 0].reshape(template.shape[0], -1)
        if self.style_template:
            obj_embedding = self.obj_vector(one_hot)  # (1, feature_dim)
        else:
            obj_embedding = self.obj_vector(one_hot)  # (1, feature_dim)
        if self.emo_type == 1:
            emo_embedding = self.emo_vector(emo)
            obj_embedding = obj_embedding + emo_embedding

        if vertice_init is not None:
            vertice_init = vertice_init.reshape(
                vertice_init.shape[0], vertice_init.shape[1], -1
            )
            start_num = vertice_init.shape[1]
        else:
            start_num = 0
        frame_num = audio.shape[1]

        if self.emo_type == 3:
            hidden_states = self.audio_emo_former(audio, emo)
        elif self.emo_type == 2:
            hidden_states = self.audio_former(audio, emo)
        else:
            hidden_states = self.audio_feature_map(audio)
        if start_num >= frame_num:
            return None
        for i in range(start_num, frame_num):
            if i == start_num:
                vertice_emb = obj_embedding.unsqueeze(1)  # (1,1,feature_dim)
                style_emb = vertice_emb
                if vertice_init is not None:
                    for j in range(vertice_init.shape[1]):
                        new_output = self.vertice_map(vertice_init[:, j, :]).unsqueeze(
                            1
                        )
                        new_output = new_output + style_emb
                        vertice_emb = torch.cat((vertice_emb, new_output), 1)

            vertice_input = self.PPE(vertice_emb)

            tgt_mask = (
                self.biased_mask[:, : vertice_input.shape[1], : vertice_input.shape[1]]
                .clone()
                .detach()
                .to(device=self.device)
            )
            tgt_mask = tgt_mask.repeat(vertice_input.shape[0], 1, 1)
            memory_mask = enc_dec_mask(
                self.device, vertice_input.shape[1], hidden_states.shape[1]
            )

            vertice_out = self.transformer_decoder(
                vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask
            )
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out.reshape(
            vertice_out.shape[0], vertice_out.shape[1], 3, -1
        )
        return vertice_out


class SpeakingStyleEncoder(nn.Module):
    def __init__(self, args):
        super(SpeakingStyleEncoder, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.n_head = 4
        self.T = args.style_T
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        self.PPE = TicPositionalEncoding(args.feature_dim, T=self.T)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=args.feature_dim,
            nhead=self.n_head,
            dim_feedforward=2 * args.feature_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.style_dim)

        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
            )

        return outputs

    def forward(self, vertice, teacher_forcing=False):
        vertice = vertice.reshape(vertice.shape[0], vertice.shape[1], -1)
        frame_num = vertice.shape[1]

        vertice_input = self.vertice_map(
            vertice.reshape(vertice.shape[0] * frame_num, -1)
        )
        vertice_input = vertice_input.reshape(vertice.shape[0], frame_num, -1)

        vertice_input = self.PPE(vertice_input)
        vertice_output = self.transformer_encoder(vertice_input)
        vertice_output = self.merged_strategy(vertice_output, mode="mean")
        style_emb = self.vertice_map_r(vertice_output)

        return style_emb


class EmotionAudioConditionEncoder(nn.Module):
    def __init__(self, args):
        super(EmotionAudioConditionEncoder, self).__init__()
        self.n_head = 4
        self.out_dim = args.feature_dim
        self.audio_feature_map = nn.Linear(512, args.feature_dim)
        self.emo_feature_map = nn.Linear(args.emo_num, args.feature_dim)
        self.PPE = TicPositionalEncoding(args.feature_dim, T=args.T)
        self.biased_mask = init_biased_mask(
            n_head=self.n_head, max_seq_len=600, T=args.T
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.feature_dim,
            nhead=self.n_head,
            dim_feedforward=2 * args.feature_dim,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.device = args.device

        self.map_output = nn.Linear(512, self.out_dim)
        self.audio_map_r = nn.Linear(args.feature_dim, 512)

        nn.init.constant_(self.map_output.weight, 0)
        nn.init.constant_(self.map_output.bias, 0)

    def forward(self, audio, emo, teacher_forcing=False):
        frame_num = audio.shape[1]
        hidden_states = self.audio_feature_map(audio)
        obj_emb = self.emo_feature_map(emo)

        if teacher_forcing:
            audio_input = obj_emb.unsqueeze(1).repeat(1, frame_num, 1)
            audio_input = self.PPE(audio_input)
            tgt_mask = (
                self.biased_mask[:, : audio_input.shape[1], : audio_input.shape[1]]
                .clone()
                .detach()
                .to(device=self.device)
            )
            memory_mask = enc_dec_mask(
                self.device, audio_input.shape[1], hidden_states.shape[1]
            )
            audio_out = self.transformer_decoder(
                audio_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask
            )
            audio_out = self.audio_map_r(audio_out)
            out = self.map_output(audio_out)
        else:
            for i in range(frame_num):
                if i == 0:
                    audio_emb = obj_emb.unsqueeze(1)
                    emo_emb = audio_emb
                    audio_input = self.PPE(emo_emb)
                else:
                    audio_input = self.PPE(audio_emb)
                tgt_mask = (
                    self.biased_mask[:, : audio_input.shape[1], : audio_input.shape[1]]
                    .clone()
                    .detach()
                    .to(device=self.device)
                )
                tgt_mask = tgt_mask.repeat(audio_input.shape[0], 1, 1)
                memory_mask = enc_dec_mask(
                    self.device, audio_input.shape[1], hidden_states.shape[1]
                )
                audio_out = self.transformer_decoder(
                    audio_input,
                    hidden_states,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                )
                audio_out = self.audio_map_r(audio_out)
                new_output = self.audio_feature_map(audio_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + emo_emb
                audio_emb = torch.cat((audio_emb, new_output), 1)

            out = self.map_output(audio_out)
        return out


class Classifier(nn.Module):
    """Classification head for temporal audio features."""

    def __init__(self, input_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(input_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AudioConditionEncoder(nn.Module):
    def __init__(self, args):
        super(AudioConditionEncoder, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.n_head = 4
        self.T = args.T
        self.output_dim = args.feature_dim
        self.vertice_map = nn.Linear(512, args.feature_dim)
        self.PPE = TicPositionalEncoding(args.feature_dim, T=self.T)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=args.feature_dim,
            nhead=self.n_head,
            dim_feedforward=2 * args.feature_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.vertice_map_r = nn.Linear(args.feature_dim, self.output_dim)

        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)
        self.emo_vector = nn.Linear(args.emo_num, args.feature_dim)

    def forward(self, audio, emo=None):
        audio = audio.reshape(audio.shape[0], audio.shape[1], -1)
        frame_num = audio.shape[1]

        audio_input = self.vertice_map(audio.reshape(audio.shape[0] * frame_num, -1))
        audio_input = audio_input.reshape(audio.shape[0], frame_num, -1)
        audio_input = self.PPE(audio_input)
        audio_output = self.transformer_encoder(audio_input)
        if emo is not None:
            emo_embedding = self.emo_vector(emo)
            emo_embedding = emo_embedding.unsqueeze(1).repeat(1, frame_num, 1)
            audio_output = audio_output + emo_embedding

        output = self.vertice_map_r(
            audio_output.reshape(audio_output.shape[0] * audio_output.shape[1], -1)
        )
        output = output.reshape(audio_output.shape[0], audio_output.shape[1], -1)

        return output
