import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

from models.base import WSConv2d, ScaledStdConv2d

from functools import partial

__all__ = ['nf_resnet32']

_nonlin_gamma = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    silu=1.7881293296813965,
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)

ignore_inplace = ['gelu', 'silu', 'softplus', ]

activation_fn = {
    'identity': lambda x, *args, **kwargs: nn.Identity(*args, **kwargs)(x) * _nonlin_gamma['identity'],
    'celu': lambda x, *args, **kwargs: nn.CELU(*args, **kwargs)(x) * _nonlin_gamma['celu'],
    'elu': lambda x, *args, **kwargs: nn.ELU(*args, **kwargs)(x) * _nonlin_gamma['elu'],
    'gelu': lambda x, *args, **kwargs: nn.GELU(*args, **kwargs)(x) * _nonlin_gamma['gelu'],
    'leaky_relu': lambda x, *args, **kwargs: nn.LeakyReLU(*args, **kwargs)(x) * _nonlin_gamma['leaky_relu'],
    'log_sigmoid': lambda x, *args, **kwargs: nn.LogSigmoid(*args, **kwargs)(x) * _nonlin_gamma['log_sigmoid'],
    'log_softmax': lambda x, *args, **kwargs: nn.LogSoftmax(*args, **kwargs)(x) * _nonlin_gamma['log_softmax'],
    'relu': lambda x, *args, **kwargs: nn.ReLU(*args, **kwargs)(x) * _nonlin_gamma['relu'],
    'relu6': lambda x, *args, **kwargs: nn.ReLU6(*args, **kwargs)(x) * _nonlin_gamma['relu6'],
    'selu': lambda x, *args, **kwargs: nn.SELU(*args, **kwargs)(x) * _nonlin_gamma['selu'],
    'sigmoid': lambda x, *args, **kwargs: nn.Sigmoid(*args, **kwargs)(x) * _nonlin_gamma['sigmoid'],
    'silu': lambda x, *args, **kwargs: nn.SiLU(*args, **kwargs)(x) * _nonlin_gamma['silu'],
    'softplus': lambda x, *args, **kwargs: nn.Softplus(*args, **kwargs)(x) * _nonlin_gamma['softplus'],
    'tanh': lambda x, *args, **kwargs: nn.Tanh(*args, **kwargs)(x) * _nonlin_gamma['tanh'],
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1,
            base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return base_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Conv2d:
    """1x1 convolution"""
    return base_conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            alpha: float = 0.2,
            beta: float = 1.0,
            activation: str = 'relu',
            base_conv: nn.Conv2d = ScaledStdConv2d
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, base_conv=base_conv)
        self.activation = activation

        if activation not in ignore_inplace:
            self.act = partial(activation_fn[activation], inplace=True)
        else:
            self.act = partial(activation_fn[activation])
        self.conv2 = conv3x3(planes, planes, base_conv=base_conv)
        self.downsample = downsample
        self.stride = stride
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = activation_fn[self.activation](x=x) * self.beta
        out = self.conv1(out)
        out = self.act(x=out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out *= self.alpha
        out += identity
        return out


class NFResNet(nn.Module):

    def __init__(
            self,
            block: Type[BasicBlock],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            alpha: float = 0.2,
            activation: str = 'relu',
            base_conv: nn.Conv2d = ScaledStdConv2d
    ) -> None:
        super(NFResNet, self).__init__()

        assert activation in activation_fn.keys()

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.initial_conv = base_conv(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1, expected_std = self._make_layer(block, 16, layers[0],
                                                     alpha=alpha, init_expected_std=1.0,
                                                     activation=activation, base_conv=base_conv)
        self.layer2, expected_std = self._make_layer(block, 32, layers[1],
                                                     stride=2, dilate=replace_stride_with_dilation[0],
                                                     alpha=alpha, init_expected_std=expected_std,
                                                     activation=activation, base_conv=base_conv)
        self.layer3, expected_std = self._make_layer(block, 64, layers[2],
                                                     stride=2, dilate=replace_stride_with_dilation[1],
                                                     alpha=alpha, init_expected_std=expected_std,
                                                     activation=activation, base_conv=base_conv)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.head_var = 'fc'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, alpha: float = 0.2, init_expected_std: float = 1.,
                    activation: str = 'relu', base_conv: nn.Conv2d = ScaledStdConv2d) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,
                        stride, base_conv=base_conv),
            )

        layers = []
        beta = 1. / init_expected_std
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, alpha=alpha, beta=beta, activation=activation,
                            base_conv=base_conv))
        expected_std = 1.0
        expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            beta = 1. / expected_std
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                alpha=alpha, beta=beta, activation=activation,
                                base_conv=base_conv))
            expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

        return nn.Sequential(*layers), expected_std

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _nf_resnet(
        arch: str,
        block: Type[BasicBlock],
        layers: List[int],
        alpha: float,
        activation: str,
        base_conv: nn.Conv2d,
        **kwargs: Any
) -> NFResNet:
    model = NFResNet(block, layers, alpha=alpha, activation=activation, base_conv=base_conv, **kwargs)
    return model


def nf_resnet32(alpha: float = 0.2, activation: str = 'relu', base_conv: nn.Conv2d = ScaledStdConv2d,
                **kwargs: Any) -> NFResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `"High-Performance Large-Scale Image Recognition Without Normalization" <https://arxiv.org/pdf/2102.06171v1>`.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _nf_resnet('resnet32', BasicBlock, [5, 5, 5], alpha=alpha, activation=activation,
                      base_conv=base_conv, **kwargs)
