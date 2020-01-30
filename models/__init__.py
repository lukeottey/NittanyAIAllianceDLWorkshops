from .resnet import *
from .vgg import *
from .alexnet import *
from .wideresnet import *
from .densenet import *


class ModelKwargs:
    def __init__(self, arguments, num_classes):
        self.kwargs = ModelKwargs.__getattribute__(self, arguments.model)(arguments, num_classes)

    def preactresnet(self, arguments, num_classes):
        return {'num_classes': num_classes, 
                'in_planes': 1 if 'mnist' in arguments.dataset else 3,
                'arch_config': arguments.arch_config}

    def resnet(self, arguments, num_classes):
        return {'num_classes': num_classes,
                'in_planes': 1 if 'mnist' in arguments.dataset else 3, 
                'arch_config': arguments.arch_config}
    
    def wideresnet(self, arguments, num_classes):
        return {'num_classes': num_classes, 'in_planes': 1 if 'mnist' in arguments.dataset else 3,
                'depth': arguments.depth, 'widen_factor': arguments.widen_factor,
                'dropout': arguments.dropout, 'pool_kernel': arguments.pool_kernel}

    def densenet(self, arguments, num_classes):
        return {'num_classes': num_classes, 'in_planes': 1 if 'mnist' in arguments.dataset else 3,
                'arch_config': arguments.arch_config}    

    def vgg(self, arguments, num_classes):
        return {'num_classes': num_classes, 'in_planes': 1 if 'mnist' in arguments.dataset else 3,
                'arch_config': arguments.arch_config}

    def alexnet(self, arguments, num_classes):
        return {'num_classes': num_classes, 'in_planes': 1 if 'mnist' in arguments.dataset else 3,
                'input_size': 'large' if arguments.dataset == 'imagenet' else 'small'}
