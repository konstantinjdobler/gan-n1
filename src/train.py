import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms

numberOfEpochs = 
batchSize = 
numWorkers = 
imageSize= 

dataset = torchvision.datasets.ImageFolder(
    root = "someFolderPath",
    transform=transforms.Compose([
        transform.Scale(imageSize),
        transforms.ToTensor()
    ])
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)

#load and init models

#set loss function and optimizer

for epoch in range(numberOfEpochs):
    for i, data in enumerate(dataloader, 0):
        #Update discriminator

        #Update generator
