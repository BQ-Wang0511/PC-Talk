"""Audio feature encoder shared by LAC and EMC."""

from torch import nn


class Conv2d(nn.Module):
    def __init__(
        self,
        cin,
        cout,
        kernel_size,
        stride,
        padding,
        residual=False,
        leakyReLU=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding), nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioVisualEncoder(nn.Module):
    def __init__(self):
        super(AudioVisualEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        input_size = x.size()
        if len(input_size) > 4:
            input = self.tensor5to4_audio(x)
        else:
            input = x

        out = self.audio_encoder(input)
        out = out.squeeze(2).squeeze(2)

        if len(input_size) > 4:
            out = out.reshape(input_size[0], input_size[1], -1)
        return out

    def tensor5to4_audio(self, input):
        if input is None:
            return None
        input_dim_size = len(input.size())
        if input_dim_size > 4:
            b, t, c, h, w = input.size()
            input = input.reshape(-1, c, h, w)
        return input
