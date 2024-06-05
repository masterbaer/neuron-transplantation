import torch.nn as nn
import torch.nn.functional as F
import torch

'''
The models used in this experiment are defined here. We use an MLP, LeNet, VGG11 and Resnet18.
For CNNs we exclude the BatchNorm2D layers and we exclude all biases to be compatible with Optimal-Transport-Fusion.
'''

def get_model_from_name(model_name: str,
                        output_dim: int = 10,
                        input_dim: int = None,
                        dataset_name: str = None,
                        bias: bool = False) -> torch.nn.Module:
    if model_name == "smallnn":
        return NeuralNetwork(input_dim, output_dim, bias=bias)

    if model_name == "lenet":
        if dataset_name == "cifar10":
            return LeNet(num_classes=output_dim, bias=bias)
        if dataset_name == "mnist":
            return LeNetMNIST(bias=False)
        if dataset_name == "svhn":
            return LeNet(num_classes=output_dim, bias=bias)

    if model_name == "vgg11":
        if dataset_name == "cifar10":
            # return VGG("VGG11", output_dim, bias=bias)

            # VGG from OT code. This has no batchnorms/biases as OT does not support those.
            return VGGOT("VGG11")
        if dataset_name == "cifar100":
            return VGGOT("VGG11", num_classes=100)

        if dataset_name == "mnist":
            exit()
        if dataset_name == "svhn":
            exit()

    if model_name == "resnet18":
        if dataset_name == "cifar10":
            return ResNet18(num_classes=output_dim, bias=bias)

            # resnet without Batchnorm / biases from the OT paper. This does not work with their OT-weight-mode
            # (only activation mode)
            # return ResNet18OT()
        if dataset_name == "mnist":
            print("not supported yet")
            exit()
        if dataset_name == "svhn":
            print("not supported yet")
            exit()
        if dataset_name == "cifar100":
            return ResNet18(num_classes=100, bias=bias)
        if dataset_name == "imagenet":
            print("todo")
            exit()

    if model_name == "vit":
        # cifar version from here: https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
        if dataset_name == "cifar10":
            return ViT(
                image_size=32,
                patch_size=4,
                num_classes=10,
                dim=512,
                depth=6,
                heads=8,
                mlp_dim=512,
                dropout=0.1,
                emb_dropout=0.1
            )
        if dataset_name == "cifar100":
            return ViT(
                image_size=32,
                patch_size=4,
                num_classes=100,
                dim=512,
                depth=6,
                heads=8,
                mlp_dim=512,
                dropout=0.1,
                emb_dropout=0.1
            )
        if dataset_name == "svhn":
            print("not supported")
            exit()

    # default: small mlp
    return NeuralNetwork(input_dim, output_dim, bias)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512, bias)
        self.fc2 = nn.Linear(512, 512, bias)
        self.fc3 = nn.Linear(512, 512, bias)
        self.fc4 = nn.Linear(512, output_dim, bias)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)

        # logits = self.linear_relu_stack(x)
        return x  # logits


class CombinedNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512 * 4)
        self.fc2 = nn.Linear(512 * 4, 512 * 4)
        self.fc3 = nn.Linear(512 * 4, 512 * 4)
        self.fc4 = nn.Linear(512 * 4, output_dim)
        # divide fc outputs by 4 when creating the model

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class AdaptiveNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1, hidden2, hidden3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class AdaptiveNeuralNetwork2(nn.Module):
    def __init__(self, input_dim, output_dim, layer_width, num_layers):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_dim, layer_width, bias=False)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_width, layer_width, bias=False) for _ in range(num_layers - 1)
        ])

        self.output_layer = nn.Linear(layer_width, output_dim, bias=False)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.input_layer(x))

        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))

        x = self.output_layer(x)
        return x


# https://github.com/soapisnotfat/pytorch-cifar10/blob/master/models/LeNet.py
class LeNet(nn.Module):
    def __init__(self, num_classes=10, bias=True):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6,
                               kernel_size=5,
                               bias=bias)  # 3 input channels,6 output feature maps, kernelsize 5 (stride 1, padding 0 implicitly)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias=bias)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=bias)
        self.fc2 = nn.Linear(120, 84, bias=bias)
        self.fc3 = nn.Linear(84, num_classes, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# adapted from https://medium.com/@deepeshdeepakdd2/lenet-5-implementation-on-mnist-in-pytorch-c6f2ee306e37
# changed tanh to relu and avg to max pooling
class LeNetMNIST(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.feature = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=bias),
            # 28*28->32*32-->28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14*14

            # 2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=bias),  # 10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 5*5

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120, bias=bias),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84, bias=bias),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10, bias=bias),
        )

    def forward(self, x):
        return self.classifier(self.feature(x))


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, bias=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], bias=bias)
        self.classifier = nn.Linear(512, num_classes, bias=bias)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, bias=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=bias),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# resnet, see https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bias=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bias=False))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bias=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def ResNet18(num_classes=10, bias=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# Resnet for MNIST from
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class BasicBlockMNIST(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bias=False):
        super(BasicBlockMNIST, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetMNIST(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale, bias=False):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits  # logits, probas


def resnet18MNIST(bias=False):
    """Constructs a ResNet-18 model."""
    model = ResNetMNIST(block=BasicBlockMNIST,
                        layers=[2, 2, 2, 2],
                        num_classes=10,
                        grayscale=True,
                        bias=bias)
    return model


# Resnet18 for SVHN:
# -------------------------

class BasicBlockSVHN(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockSVHN, self).__init__()

        resblock_out_channels = out_channels * BasicBlockSVHN.expansion
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, resblock_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(resblock_out_channels)
        )

        if stride == 1 and in_channels == resblock_out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, resblock_out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(resblock_out_channels),
            )

        self.final = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.final(self.residual(x) + self.shortcut(x))


class BottleNeckSVHN(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        resblock_out_channels = out_channels * BottleNeckSVHN.expansion
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, resblock_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(resblock_out_channels)
        )

        if stride == 1 and in_channels == resblock_out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, resblock_out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(resblock_out_channels),
            )

        self.final = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.final(self.residual(x) + self.shortcut(x))


class ResNetSVHN(nn.Module):
    def __init__(self, block_type, num_block, num_classes, **kwargs):
        super(ResNetSVHN, self).__init__()

        if block_type not in ['BasicBlock', 'BottleNeck']:
            raise NotImplementedError('Invalid block type.')

        block = BasicBlockSVHN if block_type == 'BasicBlock' else BottleNeckSVHN
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.cur_channels = 64

        self.layer2 = self._make_layer(block, 64, num_block[0], 1)
        self.layer3 = self._make_layer(block, 128, num_block[1], 2)
        self.layer4 = self._make_layer(block, 256, num_block[2], 2)
        self.layer5 = self._make_layer(block, 512, num_block[3], 2)

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(512 * block.expansion, num_classes)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            cur_stride = stride if i == 0 else 1
            layers.append(block(self.cur_channels, out_channels, cur_stride))
            self.cur_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.final(x)

    def loss(self, res, gt):
        return F.cross_entropy(res, gt)


def resnet18SVHN(**kwargs):
    return ResNetSVHN('BasicBlock', [2, 2, 2, 2], **kwargs)


###########
# Resnet18 from OT paper
class BasicBlockOT(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=False):
        super(BasicBlockOT, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckOT(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=False):
        super(BottleneckOT, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = self.bn3 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetOT(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=False, linear_bias=False):
        super(ResNetOT, self).__init__()
        self.in_planes = 64
        self.use_batchnorm = use_batchnorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Sequential()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes, bias=linear_bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18OT(num_classes=10, use_batchnorm=False, linear_bias=False):
    return ResNetOT(BasicBlockOT, [2, 2, 2, 2], num_classes=num_classes, use_batchnorm=use_batchnorm,
                    linear_bias=linear_bias)


cfgOT = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11_quad': [64, 'M', 512, 'M', 1024, 1024, 'M', 2048, 2048, 'M', 2048, 512, 'M'],
    'VGG11_doub': [64, 'M', 256, 'M', 512, 512, 'M', 1024, 1024, 'M', 1024, 512, 'M'],
    'VGG11_half': [64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGOT(nn.Module):
    def __init__(self, vgg_name, num_classes=10, batch_norm=False, bias=False, relu_inplace=True):
        super(VGGOT, self).__init__()
        self.batch_norm = batch_norm
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name], relu_inplace=relu_inplace)
        self.classifier = nn.Linear(512, num_classes, bias=self.bias)
        print("Relu Inplace is ", relu_inplace)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, relu_inplace=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.batch_norm:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=relu_inplace)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
                               nn.ReLU(inplace=relu_inplace)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        # print("in _make_layers", list(layers))
        return nn.Sequential(*layers)


####################################
# VIT from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
####################################

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
