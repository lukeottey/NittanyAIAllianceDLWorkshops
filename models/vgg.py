import torch.nn as nn

__all__ = ['vgg']

class VGG(nn.Module):
    def __init__(self, num_classes, configuration, in_planes):
        super(VGG, self).__init__()
        self.in_planes = in_planes
        if in_planes == 1:
            self.conv_transpose = nn.Sequential(
                nn.ConvTranspose2d(in_planes, out_channels=1, kernel_size=5),
                nn.ReLU(inplace=True))
        self.features = self._make_layers(configuration, in_planes)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        if self.in_planes == 1:
            x = self.conv_transpose(x)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg, in_planes):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_planes, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_planes = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)

class VGGArchitectures:
    @staticmethod
    def vgg11(num_classes, in_planes):
        configuration = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        return VGG(num_classes=num_classes, configuration=configuration, in_planes=in_planes)
    @staticmethod
    def vgg13(num_classes, in_planes):
        configuration = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        return VGG(num_classes=num_classes, configuration=configuration, in_planes=in_planes)
    @staticmethod
    def vgg16(num_classes, in_planes):
        configuration = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        return VGG(num_classes=num_classes, configuration=configuration, in_planes=in_planes)
    @staticmethod
    def vgg19(num_classes, in_planes):
        configuration = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        return VGG(num_classes=num_classes, configuration=configuration, in_planes=in_planes)

def vgg(num_classes, in_planes, arch_config):
    options = VGGArchitectures()
    return options.__getattribute__('vgg{}'.format(arch_config))(num_classes, in_planes)
