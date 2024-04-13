from load_model import load_checkpoint
from CustomDataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from generate_csv import generate_csv
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle('./data/cifar_test_nolabels.pkl')
comp_test_data = data[b'data']
ids = data[b'ids']
transform_test = transforms.Compose([
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# for i in range(20):
#     print(comp_test_data[i])
# Assuming 'your_data_lists' is the list of lists you mentioned
dataset = CustomDataset(comp_test_data, transform=transform_test)
# print(dataset[0])
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

model = load_checkpoint("./best_model/ckpt.pth")
# Assuming 'model' is your trained model
model.eval()  # Set the model to evaluation mode

predictions = []
with torch.no_grad():
    for images in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())

generate_csv(ids, predictions)
