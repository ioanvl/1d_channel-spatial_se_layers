import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSELayer1d(nn.Module):
    def __init__(self, num_channels, reduction_ratio=4):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer1d, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.activ_1 = nn.ReLU()
        self.activ_2 = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H)
        :return: output tensor
        """
        batch_size, num_channels, H = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.activ_1(self.fc1(squeeze_tensor))
        fc_out_2 = self.activ_2(self.fc2(fc_out_1))

        # a, b = squeeze_tensor.size()
        # output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1))
        return input_tensor * fc_out_2.view(batch_size, num_channels, 1)


class SpatialSELayer1d(nn.Module):

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer1d, self).__init__()
        self.conv = nn.Conv1d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        return input_tensor * squeeze_tensor.view(batch_size, 1, a)


class ChannelSpatialSELayer1d(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer1d, self).__init__()
        self.cSE = ChannelSELayer1d(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer1d(num_channels)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, W)
        :return: output_tensor
        """

        return torch.max(self.cSE(input_tensor), self.sSE(input_tensor))


"""
credit: ai-med: https://github.com/ai-med/squeeze_and_excitation

Squeeze and Excitation Module
*****************************

Collection of squeeze and excitation classes where each can be inserted as a block into a neural network architechture

    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_

"""
