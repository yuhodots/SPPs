import torchsummary
from models.resnet32 import resnet32
from models.nf_resnet32 import nf_resnet32


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    torchsummary.summary(resnet32(num_classes=100), (3, 32, 32), device='cpu')
    torchsummary.summary(nf_resnet32(activation='relu', num_classes=100), (3, 32, 32), device='cpu')


if __name__ == "__main__":
    main()
