import torchsummary
from models.resnet import resnet101
from timm.models import nf_resnet101


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    torchsummary.summary(resnet101(num_classes=100), (3, 224, 224), device='cpu')
    torchsummary.summary(nf_resnet101(num_classes=100), (3, 224, 224), device='cpu')


if __name__ == "__main__":
    main()
