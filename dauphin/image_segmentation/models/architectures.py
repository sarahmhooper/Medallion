import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from emmental.modules.identity_module import IdentityModule

"""
Unet code courtesy of https://github.com/milesial/Pytorch-UNet codebase.

This NN is modeled after the original Unet paper. 

Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for
Biomedical Image Segmentation.
Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer,
LNCS, Vol.9351: 234--241, 2015


"""

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, no_classifier=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.no_classifier = no_classifier
        self.output_dim = 64

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        if not self.no_classifier:
            self.outc = OutConv(self.output_dim, n_classes)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if not self.no_classifier:
            x = self.outc(x)
        return x

def UNet_model(
    n_channels=1, 
    n_classes=2, 
    bilinear=True, 
    no_classifier=False,
):
    """Constructs a UNet model."""
    model = UNet(
        n_channels, 
        n_classes, 
        bilinear, 
        no_classifier,
    )
    return model



"""
Gadgetron Unet code courtesy of Gadgetron open source codebase.

This NN is modified from original Unet paper, and combined good points from Vnet paper
and ResUnet paper

Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for
Biomedical Image Segmentation.
Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer,
LNCS, Vol.9351: 234--241, 2015

Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural
Networks for Volumetric Medical Image Segmentation. MICCAI, 2016.

Zhengxin Zhang, Qingjie Liu, Yunhong Wang. Road Extraction by Deep Residual U-Net.
IEEE GEOSCIENCE AND REMOTE SENSING LETTERS, 2017.

"""

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def compute_output_size_2d(m, H, W):
    """
    Compute output size for conv and pooling layer

    Input:
    - m: module, can be Conv2d or MaxPool2d or AvePool2d
    - H, W: inpute 2d image size [N C H W]

    Output:
    - H_out, W_out: output size
    """

    H_out = H
    W_out = W

    if isinstance(m, nn.Conv2d):
        kernel_size = m.kernel_size
        stride = m.stride
        padding = m.padding
        dilation = m.dilation

        H_out = (
            int(
                (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
                / stride[0]
            )
            + 1
        )
        W_out = (
            int(
                (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
                / stride[1]
            )
            + 1
        )
    elif isinstance(m, nn.modules.pooling.AvgPool2d):
        kernel_size = m.kernel_size
        stride = m.stride
        padding = m.padding

        H_out = int((H + 2 * padding - kernel_size) / stride) + 1
        W_out = int((W + 2 * padding - kernel_size) / stride) + 1
    else:
        raise Exception("Unsupport module type ... ")

    return H_out, W_out


def compute_output_size_maxpool2d(m, H, W):
    """
    Compute output size for max pooling layer

    Input:
    - m: module, can be Conv2d or MaxPool2d or AvePool2d
    - H, W: inpute 2d image size [N C H W]

    Output:
    - H_out, W_out: output size
    """

    H_out = H
    W_out = W

    kh = m.kernel_size[0]
    kw = m.kernel_size[1]
    ph = m.padding[0]
    pw = m.padding[1]

    if m.stride[0] == 2:
        H_out = int((H + 2 * ph - kh) / 2) + 1
        W_out = int((W + 2 * pw - kw) / 2) + 1
    else:
        H_out = H + 2 * ph - kh + 1
        W_out = W + 2 * pw - kw + 1

    return H_out, W_out


class GadgetronResUnetInputBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        use_dropout=False,
        p=0.5,
        H=256,
        W=256,
        verbose=True,
    ):
        super(GadgetronResUnetInputBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.use_dropout = use_dropout

        H_conv1, W_conv1 = compute_output_size_2d(self.conv1, H, W)
        H_conv2, W_conv2 = compute_output_size_2d(self.conv2, H_conv1, W_conv1)

        self.H_out = H_conv2
        self.W_out = W_conv2

        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")

        self.num_conv2d = 2

        if verbose:
            print(
                "    GadgetronResUnetInputBlock : input size (%d, %d, %d), output size "
                "(%d, %d, %d) --> (%d, %d, %d)"
                % (inplanes, H, W, planes, H_conv1, W_conv1, planes, H_conv2, W_conv2)
            )

    def forward(self, x):
        out = self.conv2(self.relu(self.bn1(self.conv1(x))))

        if self.use_dropout:
            out = self.dp(out)

        return out


class GadgetronResUnetBasicBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        use_dropout=False,
        p=0.5,
        H=256,
        W=256,
        verbose=True,
    ):
        super(GadgetronResUnetBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.dp = nn.Dropout2d(p=p)

        self.use_dropout = use_dropout
        self.dropout_p = p
        self.stride = stride

        H_conv1, W_conv1 = compute_output_size_2d(self.conv1, H, W)
        H_conv2, W_conv2 = compute_output_size_2d(self.conv2, H_conv1, W_conv1)

        self.H_out = H_conv2
        self.W_out = W_conv2

        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")

        self.num_conv2d = 2

        if verbose:
            print(
                "        GadgetronResUnetBasicBlock : input size (%d, %d, %d), output "
                "size (%d, %d, %d) --> (%d, %d, %d)"
                % (inplanes, H, W, planes, H_conv1, W_conv1, planes, H_conv2, W_conv2)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if x.shape[1] == out.shape[1]:
            out += x

        if self.use_dropout:
            out = self.dp(out)

        return out


class GadgetronResUnet_UpSample(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_ch,
        out_ch,
        bilinear=True,
        stride=1,
        use_dropout=False,
        p=0.5,
        H=256,
        W=256,
        verbose=True,
    ):
        super(GadgetronResUnet_UpSample, self).__init__()

        self.bilinear = bilinear
        if self.bilinear:
            self.up = []
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        H_layer_in = 2 * H
        W_layer_in = 2 * W

        if verbose:
            print(
                "        GadgetronResUnet_UpSample : input size (%d, %d, %d), "
                "upsampled size (%d, %d, %d)"
                % (in_ch, H, W, in_ch, H_layer_in, W_layer_in)
            )

        self.layers = layers
        self.blocks = nn.Sequential()
        self.num_conv2d = 0
        for i in range(self.layers):
            module_name = "upsample %d" % i
            if i == 0:
                self.blocks.add_module(
                    module_name,
                    block(
                        in_ch,
                        out_ch,
                        stride=stride,
                        use_dropout=use_dropout,
                        p=p,
                        H=H_layer_in,
                        W=W_layer_in,
                        verbose=verbose,
                    ),
                )
            else:
                self.blocks.add_module(
                    module_name,
                    block(
                        out_ch,
                        out_ch,
                        stride=stride,
                        use_dropout=use_dropout,
                        p=p,
                        H=H_layer_in,
                        W=W_layer_in,
                        verbose=verbose,
                    ),
                )

            H_layer_in = self.blocks._modules[module_name].H_out
            W_layer_in = self.blocks._modules[module_name].W_out
            self.num_conv2d += self.blocks._modules[module_name].num_conv2d

        self.H_out = H_layer_in
        self.W_out = W_layer_in

        self.input = None

    def forward(self, x1):
        r"""
        x1: current input
        x2: from down sample layers
        """
        x2 = self.input

        if self.bilinear:
            x1 = nn.functional.interpolate(
                x1, scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            x1 = self.up(x1)

        if x1.shape[2] < x2.shape[2] or x1.shape[3] < x2.shape[3]:
            diffX = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)

        x = self.blocks(x)

        return x


class GadgetronResUnet(nn.Module):
    def __init__(
        self,
        block,
        F0,
        inplanes,
        layers,
        layers_planes,
        use_dropout=False,
        p=0.5,
        bilinear=True,
        H=224,
        W=224,
        num_classes=10,
        no_classifier=False,
        verbose=False,
    ):
        r"""
        Implement the res-unet

        - Input:
        block: class name for basic building block, e.g. GadgetronResNetBasicBlock
        F0: number of channels in data
        inplanes: number of planes in input layer
        layers: e.g. [2 2 2], number of layers and number of basic blocks in each layer
        layers_planes: number of feature maps in each layer, [128 256 512]
        """

        super(GadgetronResUnet, self).__init__()

        self.use_dropout = use_dropout
        self.dropout_p = p
        self.bilinear = bilinear
        self.layers = layers
        self.layer_planes = layers_planes
        self.num_classes = num_classes

        if verbose:
            print("GadgetronResUnet : F0=%d, inplanes=%d" % (F0, inplanes))
            print("--" * 30)

        # --------------------------------------------------------------------
        # start up layer
        # --------------------------------------------------------------------
        self.input_layer = GadgetronResUnetInputBlock(
            F0,
            inplanes,
            stride=1,
            use_dropout=use_dropout,
            p=p,
            H=H,
            W=W,
            verbose=verbose,
        )

        input_planes = inplanes
        self.output_dim = inplanes
        self.num_conv2d = self.input_layer.num_conv2d

        if verbose:
            print("--" * 30)

        # --------------------------------------------------------------------
        # down sample layer
        # --------------------------------------------------------------------
        H_layer = self.input_layer.H_out
        W_layer = self.input_layer.W_out
        self.down_layers = nn.Sequential()
        for l in range(len(layers)):
            block_name = "Down layer %d" % l

            if verbose:
                print("    GadgetronResUnet, down layer %d:" % l)

            layer, H_layer_out, W_layer_out, num_layer_conv2d = self._make_down_layer(
                block,
                layers[l],
                inplanes,
                layers_planes[l],
                stride=1,
                H_layer=H_layer,
                W_layer=W_layer,
                verbose=verbose,
            )

            inplanes = layers_planes[l]
            H_layer, W_layer = H_layer_out, W_layer_out
            self.down_layers.add_module(block_name, layer)
            self.num_conv2d += num_layer_conv2d

        if verbose:
            print("--" * 30)

        # --------------------------------------------------------------------
        # bridge layer, still downsample along H and W, but do not increase planes
        # --------------------------------------------------------------------
        if verbose:
            print(
                "    GadgetronResUnet, bridge layer (%d, %d, %d) --> (%d, %d, %d)"
                % (
                    self.layer_planes[l],
                    H_layer,
                    W_layer,
                    self.layer_planes[l],
                    H_layer_out,
                    W_layer_out,
                )
            )
        (
            self.bridge_layer,
            H_layer_out,
            W_layer_out,
            num_layer_conv2d,
        ) = self._make_down_layer(
            block,
            layers[l],
            self.layer_planes[l],
            self.layer_planes[l],
            stride=1,
            H_layer=H_layer,
            W_layer=W_layer,
            verbose=verbose,
        )

        H_layer, W_layer = H_layer_out, W_layer_out
        self.num_conv2d += num_layer_conv2d

        if verbose:
            print("--" * 30)

        # --------------------------------------------------------------------
        # up sample layer
        # --------------------------------------------------------------------
        self.up_layers = nn.Sequential()
        for l in range(len(layers)):
            block_name = "Up layer %d" % l
            bl = len(layers) - l - 1

            if verbose:
                print("    GadgetronResUnet, up layer %d:" % l)

            if bl > 0:
                output_planes = layers_planes[bl - 1]
            else:
                output_planes = input_planes

            layer, H_layer_out, W_layer_out, num_layer_conv2d = self._make_up_layer(
                block,
                layers[bl],
                2 * layers_planes[bl],
                output_planes,
                stride=1,
                H_layer=H_layer,
                W_layer=W_layer,
                verbose=verbose,
            )

            H_layer, W_layer = H_layer_out, W_layer_out
            self.up_layers.add_module(block_name, layer)
            self.num_conv2d += num_layer_conv2d

        if verbose:
            print("    GadgetronResUnet, up layer %d:" % (l + 1))
        layer, H_layer_out, W_layer_out, num_layer_conv2d = self._make_up_layer(
            block,
            layers[0],
            2 * input_planes,
            input_planes,
            stride=1,
            H_layer=H_layer,
            W_layer=W_layer,
            verbose=verbose,
        )
        block_name = "Up layer %d" % (l + 1)
        self.up_layers.add_module(block_name, layer)
        H_layer, W_layer = H_layer_out, W_layer_out
        self.num_conv2d += num_layer_conv2d

        if verbose:
            print("--" * 30)

        # --------------------------------------------------------------------
        # output layer, 1x1 conv
        # --------------------------------------------------------------------
        self.no_classifier = no_classifier
        if not self.no_classifier:
            self.output_conv = nn.Conv2d(input_planes, num_classes, 1)
            if verbose:
                print(
                    "Output layer (%d, %d, %d) --> (%d, %d, %d)"
                    % (input_planes, H_layer, W_layer, num_classes, H_layer, W_layer)
                )
            if verbose:
                print("--" * 30)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_down_layer(
        self,
        block,
        layers,
        inplanes,
        planes,
        stride=1,
        H_layer=64,
        W_layer=64,
        verbose=True,
    ):
        layer = nn.Sequential()
        layer.add_module("downsample", nn.MaxPool2d(2))

        H_layer_in = H_layer / 2
        W_layer_in = W_layer / 2

        if verbose:
            print(
                "        GadgetronResUnet, down layer (%d, %d) -> (%d, %d)"
                % (H_layer, W_layer, H_layer_in, W_layer_in)
            )

        num_layer_conv2d = 0
        for i in range(layers):
            block_name = "ResBlock" + str(i)
            if i > 0:
                inplanes = planes
            layer.add_module(
                block_name,
                block(
                    inplanes,
                    planes,
                    stride=stride,
                    use_dropout=self.use_dropout,
                    p=self.dropout_p,
                    H=H_layer_in,
                    W=W_layer_in,
                    verbose=verbose,
                ),
            )
            num_layer_conv2d += layer._modules[block_name].num_conv2d

        H_layer_out = layer._modules[block_name].H_out
        W_layer_out = layer._modules[block_name].W_out

        return layer, H_layer_out, W_layer_out, num_layer_conv2d

    def _make_up_layer(
        self,
        block,
        layers,
        inplanes,
        planes,
        stride=1,
        H_layer=64,
        W_layer=64,
        verbose=True,
    ):
        layer = GadgetronResUnet_UpSample(
            block,
            layers,
            inplanes,
            planes,
            self.bilinear,
            stride=stride,
            use_dropout=self.use_dropout,
            p=self.dropout_p,
            H=H_layer,
            W=W_layer,
            verbose=verbose,
        )

        H_layer_out = layer.H_out
        W_layer_out = layer.W_out
        num_layer_conv2d = layer.num_conv2d

        return layer, H_layer_out, W_layer_out, num_layer_conv2d

    def forward(self, x):

        x_input = self.input_layer(x)
        num_layers = len(self.layers)

        # since the results from downsample layers are needed
        x_from_down_layers = []
        for l in range(len(self.layers)):
            if l == 0:
                x = self.down_layers[l](x_input)
            else:
                x = self.down_layers[l](x)

            x_from_down_layers.append(x)

        x = self.bridge_layer(x)

        for l in range(num_layers + 1):
            if l == num_layers:
                self.up_layers[l].input = x_input
            else:
                self.up_layers[l].input = x_from_down_layers[num_layers - l - 1]

        x = self.up_layers(x)
        if not self.no_classifier:
            x = self.output_conv(x)

        return x


def GadgetronResUnet18(
    F0=1,
    inplanes=64,
    layers=[1, 1, 1],
    layers_planes=[128, 256, 512],
    use_dropout=False,
    p=0.5,
    H=224,
    W=224,
    C=1,
    no_classifier=False,
    verbose=False,
):
    """Constructs a GadgetronResUnet model."""
    model = GadgetronResUnet(
        GadgetronResUnetBasicBlock,
        F0,
        inplanes,
        layers,
        layers_planes,
        use_dropout=use_dropout,
        p=p,
        H=H,
        W=W,
        num_classes=C,
        no_classifier=no_classifier,
        verbose=verbose,
    )
    return model





"""
RESNET CODE FROM TORCHVISION
https://pytorch.org/vision/stable/_modules/torchvision/models/segmentation/segmentation.html#fcn_resnet50
"""

from torch import nn
from typing import Any, Optional
# from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models import resnet
# from torchvision.models.segmentation import FCN#, FCNHead
from torch import Tensor
# from .._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]
        super().__init__(*layers)
        
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
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
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
    return model

def resnet50(pretrained: bool = False, progress: bool = False, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)




from collections import OrderedDict
from typing import Dict, Optional

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out["out"]

    
from collections import OrderedDict
from typing import Optional, Dict
from torch import nn, Tensor
from torch.nn import functional as F

class _SimpleSegmentationModel(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None) -> None:
        super().__init__()
#         _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
#         x = features["out"]
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result["out"]

class FCN(_SimpleSegmentationModel):
    """
    Implements FCN model from
    `"Fully Convolutional Networks for Semantic Segmentation"
    <https://arxiv.org/abs/1411.4038>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass
    
def fcn_resnet50(
    num_classes: int = 2,
    no_classifier: bool = False,
) -> nn.Module:
    
    out_layer = 'layer4'
    out_inplanes = 2048

    backbone = IntermediateLayerGetter(resnet50(replace_stride_with_dilation=[False, True, True]), 
                                       return_layers={out_layer: 'out'})

    classifier = FCNHead(out_inplanes, num_classes)
    if no_classifier:
        classifier = IdentityModule()

    return FCN(backbone, classifier, aux_classifier=None)



