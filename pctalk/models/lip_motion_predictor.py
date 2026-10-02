"""Lip-motion, speaking-style, and refinement network definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .self_attention_pooling import SelfAttentionPooling


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


class LipMotionPredictor(nn.Module):
    def __init__(self, args):
        super(LipMotionPredictor, self).__init__()
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

        if args.style_template:
            if args.style_freeze:
                args.style_dim = args.person_num
                args.style_T = 100
                self.style_predictor = SpeakingStyleEncoder3(args)
                self.obj_vector = nn.Linear(
                    args.person_num, args.feature_dim, bias=False
                )
                for name, param in self.named_parameters():
                    if "style_predictor" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                args.style_dim = args.feature_dim
                args.style_T = 100
                if args.use_arc:
                    self.obj_vector = SpeakingStyleEncoder2(args)
                else:
                    self.obj_vector = SpeakingStyleEncoder(args)
            self.use_arc = args.use_arc
            self.style_freeze = args.style_freeze
        else:
            self.obj_vector = nn.Linear(args.person_num, args.feature_dim, bias=False)
        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)
        self.style_template = args.style_template

    def forward(
        self, audio, template, vertice, one_hot, ref_style=None, teacher_forcing=False
    ):
        vertice = vertice.reshape(vertice.shape[0], vertice.shape[1], -1)
        template = template[:, 0].reshape(template.shape[0], -1)

        if self.style_template:
            if self.style_freeze:
                if self.use_arc:
                    logits, arcface_loss = self.style_predictor(ref_style, one_hot)

                else:
                    logits = self.style_predictor(ref_style)
                obj_embedding = self.obj_vector(logits)  # (1, feature_dim)
            else:
                if self.use_arc:
                    obj_embedding, arcface_loss = self.obj_vector(ref_style, one_hot)
                else:
                    obj_embedding, logits = self.obj_vector(
                        ref_style
                    )  # (1, feature_dim)
        else:
            obj_embedding = self.obj_vector(one_hot)  # (1, feature_dim)

        frame_num = vertice.shape[1]

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

        if self.style_template:
            if self.use_arc:
                return vertice_out, arcface_loss
            else:
                return vertice_out, logits
        else:
            return vertice_out

    def predict(
        self,
        audio,
        template,
        one_hot,
        ref_style=None,
        vertice_init=None,
        style_embedding=None,
    ):
        template = template[:, 0].reshape(template.shape[0], -1)
        if style_embedding is not None:
            obj_embedding = style_embedding
        elif self.style_template:
            if self.style_freeze:
                if self.use_arc:
                    logits, _ = self.style_predictor(ref_style, one_hot)
                else:
                    logits = self.style_predictor(ref_style)
                obj_embedding = self.obj_vector(logits)  # (1, feature_dim)
            else:
                if self.use_arc:
                    obj_embedding, _ = self.obj_vector(ref_style, one_hot)
                else:
                    obj_embedding, _ = self.obj_vector(ref_style)  # (1, feature_dim)
        else:
            obj_embedding = self.obj_vector(one_hot)  # (1, feature_dim)

        if vertice_init is not None:
            vertice_init = vertice_init.reshape(
                vertice_init.shape[0], vertice_init.shape[1], -1
            )
            start_num = vertice_init.shape[1]
        else:
            start_num = 0
        frame_num = audio.shape[1]
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

    def style_interpolate(
        self, audio, template, one_hot1, one_hot2, alpha=0.5, vertice_init=None
    ):
        template = template[:, 0].reshape(template.shape[0], -1)
        if self.style_template:
            obj_embedding1, logits1 = self.obj_vector(one_hot1)  # (1, feature_dim)
            obj_embedding2, logits2 = self.obj_vector(one_hot2)  # (1, feature_dim)
        else:
            obj_embedding1 = self.obj_vector(one_hot1)  # (1, feature_dim)
            obj_embedding2 = self.obj_vector(one_hot2)  # (1, feature_dim)
        obj_embedding = obj_embedding1 * (1 - alpha) + obj_embedding2 * alpha
        if vertice_init is not None:
            vertice_init = vertice_init.reshape(
                vertice_init.shape[0], vertice_init.shape[1], -1
            )
            start_num = vertice_init.shape[1]
        else:
            start_num = 0
        frame_num = audio.shape[1]
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

    def get_objvector_output(self, one_hot):
        if self.style_template:
            obj_embedding, logits = self.obj_vector(one_hot)
        else:
            obj_embedding = self.obj_vector(one_hot)
        return obj_embedding


class LipMotionPredictorFineTune(nn.Module):
    def __init__(self, args):
        super(LipMotionPredictorFineTune, self).__init__()
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
        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, audio, vertice):
        vertice = vertice.reshape(vertice.shape[0], vertice.shape[1], -1)
        hidden_states = self.audio_feature_map(audio)
        vertice_input = self.vertice_map(vertice)
        vertice_input = self.PPE(vertice_input)
        vertice_out = self.transformer_decoder(vertice_input, hidden_states)
        vertice_out = self.vertice_map_r(vertice_out)

        vertice_out = vertice_out.reshape(
            vertice_out.shape[0], vertice_out.shape[1], 3, -1
        )

        return vertice_out


class LipRefinementNetwork(nn.Module):
    def __init__(self, args):
        super(LipRefinementNetwork, self).__init__()
        self.out_fc = nn.Linear(512 + 3 * 6, 3 * 6)
        self.out_fc.weight.data.zero_()
        self.out_fc.bias.data.zero_()

    def forward(self, drv_audio_fea, src_exp_in, alpha=1.0):
        b, t = src_exp_in.shape[0], src_exp_in.shape[1]
        src_exp = src_exp_in.reshape(b * t, -1)

        drv_audio_fea = drv_audio_fea.reshape(b * t, -1)

        concat_fea = torch.concat([src_exp, drv_audio_fea], dim=1)
        delta_exp = self.out_fc(concat_fea)
        delta_exp = delta_exp.reshape(b, t, 3, -1)
        out_exp = src_exp_in + delta_exp * alpha
        return out_exp


class Classifier(nn.Module):
    """Classification head for temporal motion features."""

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

        self.cls = Classifier(input_size=args.style_dim, num_labels=args.person_num)
        self.pooling = SelfAttentionPooling(args.style_dim)

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
        elif mode == "attention":
            outputs = self.pooling(hidden_states)
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
        style_emb = self.vertice_map_r(vertice_output)
        style_emb = self.merged_strategy(style_emb, mode="attention")

        logits = self.cls(style_emb)
        return style_emb, logits


class AngularMarginLoss(nn.Module):
    """
    Angular-margin classification loss.
    Args:
        in_features: size of input features
        out_features: number of classes
        s: scale factor
        m: margin angle in radians
    """

    def __init__(self, in_features, out_features, pretrain=None, s=30.0, m=0.50):
        super(AngularMarginLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        if pretrain is not None:
            self.init_with_pretrain(pretrain)
        else:
            nn.init.xavier_uniform_(self.weight)

        self.register_buffer("cos_m", torch.tensor(torch.cos(torch.tensor(m))))
        self.register_buffer("sin_m", torch.tensor(torch.sin(torch.tensor(m))))
        self.register_buffer("th", torch.tensor(torch.cos(torch.tensor(torch.pi - m))))
        self.register_buffer(
            "mm", torch.tensor(torch.sin(torch.tensor(torch.pi - m)) * m)
        )

    def init_with_pretrain(self, pretrain):
        for i in range(self.out_features):
            one_hot = torch.zeros(self.out_features)
            one_hot[i] = 1.0
            one_hot = one_hot.unsqueeze(0)
            weight = pretrain.get_objvector_output(one_hot)
            self.weight.data[i] = weight.squeeze(0)
        self.weight.requires_grad = False

    def forward(self, features, labels):
        """
        Args:
            features: input features (batch_size, in_features)
            labels: ground truth labels (batch_size,)
        Returns:
            angular-margin loss value
        """
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)

        cosine = F.linear(features, weight)  # (batch_size, out_features)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = F.cross_entropy(output, labels)
        return loss, output


class SpeakingStyleEncoder2(nn.Module):
    def __init__(self, args):
        super(SpeakingStyleEncoder2, self).__init__()
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

        self.arcface_loss = AngularMarginLoss(
            in_features=args.style_dim, out_features=args.person_num
        )
        self.pooling = SelfAttentionPooling(args.style_dim)

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
        elif mode == "attention":
            outputs = self.pooling(hidden_states)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
            )

        return outputs

    def forward(self, vertice, one_hot):
        vertice = vertice.reshape(vertice.shape[0], vertice.shape[1], -1)
        frame_num = vertice.shape[1]
        vertice_input = self.vertice_map(
            vertice.reshape(vertice.shape[0] * frame_num, -1)
        )
        vertice_input = vertice_input.reshape(vertice.shape[0], frame_num, -1)

        vertice_input = self.PPE(vertice_input)
        vertice_output = self.transformer_encoder(vertice_input)
        vertice_output = self.merged_strategy(vertice_output, mode="attention")
        loss_arcface, _ = self.arcface_loss(vertice_output, one_hot.argmax(dim=1))

        style_emb = self.vertice_map_r(vertice_output)

        return style_emb, loss_arcface


class SpeakingStyleEncoder3(nn.Module):
    def __init__(self, args):
        super(SpeakingStyleEncoder3, self).__init__()
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

        self.outlayer = nn.Sequential(self.vertice_map_r, nn.Softmax(dim=1))

        self.pooling = SelfAttentionPooling(args.feature_dim)
        self.use_arc = args.use_arc
        if args.use_arc:
            self.arcface_loss = AngularMarginLoss(
                in_features=args.feature_dim, out_features=args.person_num
            )
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
        elif mode == "attention":
            outputs = self.pooling(hidden_states)
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
            )

        return outputs

    def forward(self, vertice, one_hot=None):
        vertice = vertice.reshape(vertice.shape[0], vertice.shape[1], -1)
        frame_num = vertice.shape[1]
        vertice_input = self.vertice_map(
            vertice.reshape(vertice.shape[0] * frame_num, -1)
        )
        vertice_input = vertice_input.reshape(vertice.shape[0], frame_num, -1)

        vertice_input = self.PPE(vertice_input)
        vertice_output = self.transformer_encoder(vertice_input)
        vertice_output = self.merged_strategy(vertice_output, mode="attention")
        if self.use_arc:
            loss_arcface, style_emb = self.arcface_loss(
                vertice_output, one_hot.argmax(dim=1)
            )
            style_emb = torch.softmax(style_emb, dim=1)
            return style_emb, loss_arcface
        else:
            style_emb = self.outlayer(vertice_output)

            return style_emb


class EmotionAudioConditionEncoder(nn.Module):
    def __init__(self, args):
        super(EmotionAudioConditionEncoder, self).__init__()
        self.n_head = 4
        self.audio_feature_map = nn.Linear(512, args.feature_dim)
        self.emo_feature_map = nn.Linear(args.emo_dim, args.feature_dim)
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

        self.map_output = nn.Linear(512, 512)
        self.audio_map_r = nn.Linear(args.feature_dim, 512)

        nn.init.constant_(self.map_r.weight, 0)
        nn.init.constant_(self.map_r.bias, 0)

    def forward(self, audio, emo, teacher_forcing=False):
        frame_num = audio.shape[1]
        audio_emb = self.audio_feature_map(audio)
        emo_emb = self.emo_feature_map(emo)
        if emo_emb.shape[1] != frame_num:
            emo_emb = emo_emb.unsqueeze(1).repeat(1, frame_num, 1)

        if teacher_forcing:
            audio_input = emo_emb
            audio_input = self.PPE(audio_input)
            tgt_mask = (
                self.biased_mask[:, : audio_input.shape[1], : audio_input.shape[1]]
                .clone()
                .detach()
                .to(device=self.device)
            )
            memory_mask = enc_dec_mask(
                self.device, audio_input.shape[1], audio_emb.shape[1]
            )
            audio_out = self.transformer_decoder(
                audio_input, audio_emb, tgt_mask=tgt_mask, memory_mask=memory_mask
            )
            audio_out = self.audio_map_r(audio_out)
            out = self.map_output(audio_out)
        else:
            for i in range(frame_num):
                if i == 0:
                    audio_input = emo_emb
                    audio_input = self.PPE(audio_input)
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
                    self.device, audio_input.shape[1], audio_emb.shape[1]
                )
                audio_out = self.transformer_decoder(
                    audio_input, audio_emb, tgt_mask=tgt_mask, memory_mask=memory_mask
                )
                audio_out = self.audio_map_r(audio_out)
                new_output = self.audio_feature_map(audio_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + emo_emb
                audio_emb = torch.cat((audio_emb, new_output), 1)
            out = self.map_output(audio_out)
        return out
