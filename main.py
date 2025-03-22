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

    def __len__(self):
        return int(len(self.entries) / 10)

    def __getitem__(self, idx):
        text = self.entries.loc[idx, 'Generation']
        label = self.entries.loc[idx, 'label']

        inputTensor = self.wordToTensor(text)
        outputTensor = self.labelToTensor(label)
        return inputTensor, outputTensor
    
    def wordToTensor(self, text):
        inputTokens = self.enc.encode(text)

        # create a input for each token
        numInputNodes = self.enc.max_token_value
        inputTensor = torch.zeros(numInputNodes)

        numOfInputTokens = len(inputTokens)
        for inputToken in inputTokens:
            # divide by number of tokens to ensure no input node is above 1
            inputTensor[inputToken] += (1 / numOfInputTokens)
        return inputTensor
    
    def labelToTensor(self, label):
        labelTensor = torch.zeros(2)
        if label == 'human':
            labelTensor[0] = 1
        else:
            labelTensor[1] = 1
        return labelTensor
        


training_dataset = CustomDataset("./data/TuringBench/TT_gpt2_small/train.csv")
testing_dataset = CustomDataset("./data/TuringBench/TT_gpt2_small/test.csv")

# Create data loaders.
train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

numInputNodes = tiktoken.get_encoding("cl100k_base").max_token_value

# train_features, train_labels = next(iter(train_dataloader))
# x : torch.Tensor = train_features
# print(x.size())
# exit(0)

inputString = "'dead sea shrinking by 1 meter every year the indian expressa new study suggests that loss of water in gulf mexico, which is about to be fully depleted, could as much twice amount was lost during great depression.the study, published journal nature g cience, shows mexico now 2.5 meters year, or roughly inch per year.advertisement continue reading main storythe these changes have an effect on ocean circulation and level mexico.these would catastrophic ecosystems said jonathan w. jorgensen, a geologist at niversity north carolina chapel hill co-author. this global warming are not bad thing. letter sign p story please verify you're robot clicking box. invalid email address. re-enter. you must select newsletter subscribe to. will receive emails containing news content, updates promotions from york times. may opt-out any time. agree occasional special offers for times's products services. thank subscribing. error has occurred. try again later. view all times newsletters.the researchers say if gulf's decline, it major event history planet.this very big fish, dr. james a. mcilroy, british geological survey. one most significant events planet.the important places planet fossil fuel industry, well its'"
enc = tiktoken.get_encoding("cl100k_base")
inputTokens = enc.encode(inputString)

# create a input for each token
numInputNodes = enc.max_token_value
inputTensor = torch.zeros(numInputNodes)

numOfInputTokens = len(inputTokens)
for inputToken in inputTokens:
    # divide by number of tokens to ensure no input node is above 1
    inputTensor[inputToken] += (1 / numOfInputTokens)

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

model = NeuralNetwork(numInputNodes).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

with torch.no_grad():
    inputTensor = inputTensor.to(device)
    outputTensor = model(inputTensor)
    predictionTensor : torch.Tensor = outputTensor.softmax(dim=0)
    predictionTensor = predictionTensor.cpu()
    predictions = predictionTensor.numpy()
    print(f"chance it was written by a human: {predictions[0]}")