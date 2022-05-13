import torchsummary
from models.resnet import resnet152
from models.nf_resnet import nf_resnet152


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    torchsummary.summary(resnet152(num_classes=100), (3, 224, 224), device='cpu')
    torchsummary.summary(nf_resnet152(activation='relu', num_classes=100), (3, 224, 224), device='cpu')


if __name__ == "__main__":
    main()
