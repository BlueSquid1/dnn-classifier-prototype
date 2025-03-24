import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd

import tiktoken

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.entries = pd.read_csv(csv_file)
        self.enc = tiktoken.get_encoding("cl100k_base")
        self._inputTensorLen = 15

    def __len__(self):
        return int(len(self.entries)/2)

    def __getitem__(self, idx):
        text = self.entries.loc[idx, 'Generation']
        label = self.entries.loc[idx, 'label']

        inputTensor = self.wordToTensor(text)
        outputTensor = self.labelToTensor(label)
        return inputTensor, outputTensor
    
    def inputTensorLength(self):
        return self._inputTensorLen
    
    def wordToTensor(self, text):
        inputTokens = self.enc.encode(text)

        intputLength = self._inputTensorLen

        # create a input for each token
        inputTensor = torch.zeros(intputLength)

        for i in range(min(intputLength, len(inputTokens))):
            # divide by number of tokens to ensure no input node is above 1
            inputTensor[i] = inputTokens[i] / self.enc.max_token_value
        return inputTensor
    
    def labelToTensor(self, label):
        labelTensor = torch.zeros(2)
        if label == 'human':
            labelTensor[1] = 1
        else:
            labelTensor[0] = 1
        return labelTensor
        


training_dataset = CustomDataset("./data/TuringBench/TT_gpt3/train.csv")
testing_dataset = CustomDataset("./data/TuringBench/TT_gpt3/test.csv")

# Create data loaders.
train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

numInputNodes = tiktoken.get_encoding("cl100k_base").max_token_value

# train_features, train_labels = next(iter(train_dataloader))
# x : torch.Tensor = train_features
# print(x.size())
# exit(0)

inputString = "'dead sea shrinking by 1 meter every year the indian expressa new study"
enc = tiktoken.get_encoding("cl100k_base")
inputTokens = enc.encode(inputString)

# create a input for each token
inputTensor = torch.zeros(training_dataset.inputTensorLength())

for i in range(training_dataset.inputTensorLength()):
    inputTensor[i] = inputTokens[i] / enc.max_token_value

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, inputNodes):
        super().__init__()

        hiddenNodes = 512
        outputNodes = 2
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputNodes, hiddenNodes),
            nn.ReLU(),
            nn.Linear(hiddenNodes, outputNodes)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

inputTensorLength = training_dataset.inputTensorLength()
model = NeuralNetwork(inputTensorLength).to(device)
#model.load_state_dict(torch.load('model.pth'))
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (input, label) in enumerate(dataloader):
        input, label = input.to(device), label.to(device)

        # Compute prediction error
        pred = model(input)
        loss = loss_fn(pred, label)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for input, label in dataloader:
            input, label = input.to(device), label.to(device)
            pred = model(input)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def verify(model, inputTensor):
    with torch.no_grad():
        inputTensor = inputTensor.to(device)
        outputTensor = model(inputTensor)
        predictionTensor : torch.Tensor = outputTensor.softmax(dim=0)
        predictionTensor = predictionTensor.cpu()
        predictions = predictionTensor.numpy()
        print(f"chance it was written by GPT: {predictions[0]}")

epochs = 40
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    verify(model, inputTensor)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")