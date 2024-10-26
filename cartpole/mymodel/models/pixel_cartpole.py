import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


class ObsEncoder(nn.Module):
    def __init__(self, input_shape, embedding_size, info):
        """
        :param input_shape: tuple containing shape of input
        :param embedding_size: Supposed length of encoded vector
        """
        super(ObsEncoder, self).__init__()
        # print("ObsEncoder input_shape: ", input_shape, "embed_size", embedding_size)
        self.shape = input_shape
        activation = nn.ReLU
        d = 32
        self.d = d
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_shape[0], d, kernel_size=4, stride=2),
            activation(),
            nn.Conv2d(d, 2 * d, kernel_size=4, stride=2),
            activation(),
            nn.Conv2d(2 * d, 4 * d, kernel_size=4, stride=2),
            activation(),
            nn.Conv2d(4 * d, 8 * d, kernel_size=4, stride=2),
            activation(),
            # nn.Conv2d(8 * d, 8 * d, kernel_size=3, stride=1),
            # activation(),
        )
        if embedding_size == self.embed_size:
            self.fc_1 = nn.Identity()
        else:
            self.fc_1 = nn.Linear(self.embed_size, embedding_size)

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        embed = self.fc_1(embed)
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, 4, 2)
        # print(conv1_shape)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, 2)
        # print(conv2_shape)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, 2)
        # print(conv3_shape)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, 2)
        # print(conv4_shape)
        # conv5_shape = conv_out_shape(conv4_shape, 0, 3, 1)
        # print(conv5_shape)
        embed_size = int(8 * self.d * np.prod(conv4_shape).item())
        return embed_size


class ObsDecoder(nn.Module):
    def __init__(self, output_shape, embed_size, info):
        """
        :param output_shape: tuple containing shape of output obs
        :param embed_size: the size of input vector, for dreamerv2 : modelstate 
        """
        super(ObsDecoder, self).__init__()
        # print("ObsDecoder output_shape: ", output_shape, "embed_size", embed_size)
        self.shape = output_shape
        c, h, w = output_shape
        activation = nn.ReLU
        d = 32

        if embed_size == 1024:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(embed_size, 1024)
        self.decoder = nn.ModuleList([
            # nn.ConvTranspose2d(1024, 4 * d, kernel_size=3, stride=1),
            # activation(),
            nn.ConvTranspose2d(1024, 4 * d, kernel_size=5, stride=2),
            activation(),
            nn.ConvTranspose2d(4 * d, 2 * d, kernel_size=5, stride=2),
            activation(),
            nn.ConvTranspose2d(2 * d, d,  kernel_size=6, stride=2),
            activation(),
            nn.ConvTranspose2d(d, c, kernel_size=6, stride=2)
        ])

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        # print("original x shape: ", x.shape)
        x = x.reshape(squeezed_size, embed_size)
        # print("x shape after reshape: ", x.shape)
        x = self.linear(x)
        # print("x shape after linear: ", x.shape)
        x = torch.reshape(x, (squeezed_size, 1024, 1, 1))
        # print("x shape after reshape: ", x.shape)
        for layer in self.decoder:
            x = layer(x)
            # print("x shape after layer: ", x.shape)
        mean = torch.reshape(x, (*batch_shape, *self.shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.shape))
        return obs_dist

    # @property
    # def emb(self):
    #     conv1_shape = conv_out_shape(self.shape[1:], 0, 6, 2)
    #     print(conv1_shape)
    #     conv2_shape = conv_out_shape(conv1_shape, 0, 6, 2)
    #     print(conv2_shape)
    #     conv3_shape = conv_out_shape(conv2_shape, 0, 5, 2)
    #     print(conv3_shape)
    #     conv4_shape = conv_out_shape(conv3_shape, 0, 5, 2)
    #     print(conv4_shape)
    #     conv5_shape = conv_out_shape(conv4_shape, 0, 5, 2)
    #     print(conv5_shape)
    #     embed_size = int(4 * self.d * np.prod(conv5_shape).item())
    #     return embed_size


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
