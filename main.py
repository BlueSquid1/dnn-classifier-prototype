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
        return 20 #nt(len(self.entries) / 10)

    def __getitem__(self, idx):
        text = self.entries.loc[idx, 'Generation']
        label = self.entries.loc[idx, 'label']

        inputTensor = self.wordToTensor(text)
        outputTensor = self.labelToTensor(label)
        return inputTensor, outputTensor
    
    def inputTensorLength(self):
        return 10
    
    def wordToTensor(self, text):
        inputTokens = self.enc.encode(text)

        # create a input for each token
        inputTensor = torch.zeros(self.inputTensorLength())

        for i in range(10):
            inputTensor[i] = inputTokens[i] / self.enc.max_token_value

        # numOfInputTokens = len(inputTokens)
        # for inputToken in inputTokens:
        #     # divide by number of tokens to ensure no input node is above 1
        #     inputTensor[inputToken] += (1 / numOfInputTokens)
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

inputString = "related. as the 15th lok sabha launched its first session today,prominent faces of previous house +0097 sonia gandhi,pranab mukherjee and l k advani were ensconced in their old seats with prime minister manmohan singh too same place. occupants rest numbers up,the congress swamped second block,pushing allies to third. rahul gandhi stayed put at a rear bench but that did not deter members from trying get closer him. lalu prasad yadav,a leading light last government,sat away congress,with his new friend mulayam yadav. flock routed ousted,lalu was no longer usual vocal self. if console him after keeping out dispensation,sonia thumped desk extra enthusiasm when he stood up take oath. taking cue,congress cheered loudly,but it failed lift lalu's spirits. singh,his sp contingent reduced nearly half,looked forlorn. third partner,ram vilas paswan,was missing,having been defeated polls. ram sunder das,89,of jd( ),who had ljp chief,sat behind party president sharad yadav,basking glory success bihar. anyone matched +0092 s wry smiles while shaking hands members,it basudeb acharia. bulk cpi(m) polls,acharia seemed have lost track scattered brood. language oath became political statement. chaste hindi for gandhi,sushma swaraj opted sanskrit. parliamentary affairs pawan kumar bansal chose punjabi,prompting ask later this indicated an impending merger chandigarh (his seat) punjab."
enc = tiktoken.get_encoding("cl100k_base")
inputTokens = enc.encode(inputString)

# create a input for each token
inputTensor = torch.zeros(training_dataset.inputTensorLength())

for i in range(10):
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
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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

def verify(model, inputTensor):
    with torch.no_grad():
        inputTensor = inputTensor.to(device)
        outputTensor = model(inputTensor)
        predictionTensor : torch.Tensor = outputTensor.softmax(dim=0)
        predictionTensor = predictionTensor.cpu()
        predictions = predictionTensor.numpy()
        print(f"chance it was written by a human: {predictions[0]}")

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    verify(model, inputTensor)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")