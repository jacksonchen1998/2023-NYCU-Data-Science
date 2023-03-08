import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torchvision.models as models
from torchsummary import summary


def test_resnet50_on_fashion_mnist(weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)

    # ResNet-50 Model class
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.resnet50 = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, 10)

        def forward(self, x):
            x = self.resnet50.conv1(x)
            x = self.resnet50.bn1(x)
            x = self.resnet50.relu(x)
            x = self.resnet50.maxpool(x)

            x = self.resnet50.layer1(x)
            x = self.resnet50.layer2(x)
            x = self.resnet50.layer3(x)
            x = self.resnet50.layer4(x)

            x = self.resnet50.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.resnet50.fc(x)
            return x

    net = ResNet().to(device)

    checkpoint = torch.load(weights_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    
    summary(net, (3, 28, 28))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the {total} test images: {accuracy:.2f} %")

    return accuracy


        
def main():
    ckpt_path ="./resnet-50.pth"
    test_resnet50_on_fashion_mnist(ckpt_path)
    print("----------------------------------------------")

if __name__ == "__main__":
    main()        