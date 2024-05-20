from torchvision.datasets import ImageFolder

if __name__ == "__main__":
    data = ImageFolder('dataset/imageNet/tiny-imagenet-200/train')

    print(data[0])
