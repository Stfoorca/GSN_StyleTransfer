import torch
import torch.nn as nn
import torch.nn.functional as F


# region SimpleModels

def simpledeconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def simpleconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class SimpleResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(SimpleResnetBlock, self).__init__()
        self.conv_layer = simpleconv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.conv_layer(x)


class SimpleDiscriminator(nn.Module):
    def __init__(self, in_c=3, conv_dim=64):
        super(SimpleDiscriminator, self).__init__()

        self.conv1 = simpleconv(in_channels=in_c, out_channels=conv_dim, kernel_size=4)
        self.conv2 = simpleconv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4)
        self.conv3 = simpleconv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4)
        self.conv4 = simpleconv(in_channels=conv_dim * 4, out_channels=1, kernel_size=4, padding=0, batch_norm=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.conv4(out).squeeze()
        out = F.sigmoid(out)

        return out


class SimpleGenerator(nn.Module):
    def __init__(self, in_c=3, conv_dim=64, init_zero_weights=False):
        super(SimpleGenerator, self).__init__()

        self.conv1 = simpleconv(in_channels=in_c, out_channels=conv_dim, kernel_size=4)
        self.conv2 = simpleconv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4)

        self.resnet_block = SimpleResnetBlock(conv_dim * 2)

        self.deconv1 = simpledeconv(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4)
        self.deconv2 = simpledeconv(in_channels=conv_dim, out_channels=3, kernel_size=4, batch_norm=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.relu(self.resnet_block(out))

        out = F.relu(self.deconv1(out))
        out = F.tanh(self.deconv2(out))

        return out


# endregion
# region ComplexModels

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    layers = []

    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    layers.append(norm_layer(out_channels))
    layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


def convL(in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.BatchNorm2d, bias=False):
    layers = []

    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    layers.append(norm_layer(out_channels))
    layers.append(nn.LeakyReLU(0.2, True))

    return nn.Sequential(*layers)


def deconv(in_channels, out_channels, kernel_size, stride=1, padding=0, out_padding=0, norm_layer=nn.BatchNorm2d,
           bias=False):
    layers = []

    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, out_padding, bias=bias))
    layers.append(norm_layer(out_channels))
    layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super(ResnetBlock, self).__init__()

        block = [nn.ReflectionPad2d(1),
                 conv(dim, dim, kernel_size=3, norm_layer=norm_layer, bias=use_bias),
                 nn.ReflectionPad2d(1),
                 nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                 norm_layer(dim)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_c, filters_n=64, layers_n=3, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Discriminator, self).__init__()

        layers = []
        use_bias = norm_layer == nn.InstanceNorm2d

        layers += nn.Conv2d(in_c, filters_n, kernel_size=4, stride=2, padding=1)
        layers += nn.LeakyReLU(0.2, True)

        filters_n_multi = 1
        filters_n_prev_multi = 1

        for i in range(1, layers_n):
            filters_n_prev_multi = filters_n_multi
            filters_n_multi = min(2 ** i, 8)
            layers += convL(filters_n * filters_n_prev_multi, filters_n * filters_n_multi, kernel_size=4, stride=2,
                            norm_layer=norm_layer, padding=1, bias=use_bias)

        filters_n_prev_multi = filters_n_multi
        filters_n_multi = min(2 ** layers_n, 8)

        layers += convL(filters_n * filters_n_prev_multi, filters_n * filters_n_multi, kernel_size=4, stride=1,
                        norm_layer=norm_layer, padding=1, bias=use_bias)
        layers += nn.Conv2d(filters_n * filters_n_multi, 1, kernel_size=4, stride=1, padding=1)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_c=3, out_c=3, filters_n=64, norm_layer=nn.BatchNorm2d, blocks_n=9):
        super(Generator, self).__init__()

        layers = []
        use_bias = norm_layer == nn.InstanceNorm2d

        layers += nn.ReflectionPad2d(3)
        layers += conv(in_c, filters_n, 7, norm_layer=norm_layer, bias=use_bias)
        layers += conv(filters_n, filters_n * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)
        layers += conv(filters_n * 2, filters_n * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)

        for i in range(blocks_n):
            layers += ResnetBlock(filters_n * 4, norm_layer, use_bias)

        layers += deconv(filters_n * 4, filters_n * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias)
        layers += deconv(filters_n * 2, filters_n, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias)
        layers += nn.ReflectionPad2d(3)
        layers += nn.Conv2d(filters_n, out_c, 7)
        layers += nn.Tanh()

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# endregion
