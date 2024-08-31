import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Model_classes_for_ensemble import ConvTransformer, CNNModel, ViT_modified

#ensemble function
def ensemble_predict(input_data):
    with torch.no_grad():
        cnn_output = cnn_model(input_data)
        vit_output = vit_model(input_data)
        hybrid_output = hybrid_model(input_data)
        
        ensemble_output = (cnn_output + vit_output + hybrid_output) / 3
        return ensemble_output
def evaluate_ensemble(test_loader):
    total = 0
    correct = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = ensemble_predict(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Ensemble Model Accuracy on CIFAR-100 Test Set: {accuracy:.2f}%')


if __name__ == '__main__':
    dim = 192
    depth = 14
    heads= 8
    mlp_dim = 2048
    num_channels = 3
    dim_head = 64 #64
    dropout = 0.0
    emb_dropout = 0.0
    patch_size = 4
    img_size = 32
    num_classes = 100

    vit_model = ViT_modified(dim = dim,
                        depth = depth,
                        heads = heads,
                        mlp_dim = mlp_dim,
                        pool = 'cls',
                        num_classes = num_classes,
                        num_channels = num_channels,
                        dim_head = dim_head,
                        dropout = dropout,
                        emb_dropout = emb_dropout,
                        img_size = img_size,
                        patch_size = patch_size
                     )
    hybrid_model = ConvTransformer()
    cnn_model = CNNModel()

    #loading the models
    cnn_model.load_state_dict(torch.load('cnn_model.pth'))
    vit_model.load_state_dict(torch.load('model3_14_8_2048_weights.pth'))
    hybrid_model.load_state_dict(torch.load('ResConvTrans.pth'))

    vit_model.eval()
    cnn_model.eval()
    hybrid_model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnn_model.to(device)
    vit_model.to(device)
    hybrid_model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    #Evaluate the ensemble
    evaluate_ensemble(test_loader)