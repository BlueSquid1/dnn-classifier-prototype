import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import tiktoken


inputString = "hello world!"
enc = tiktoken.get_encoding("cl100k_base")
inputTokens = enc.encode(inputString)

# create a input for each token
numInputNodes = enc.max_token_value
inputTensor = torch.zeros(numInputNodes)

numOfInputTokens = len(inputTokens)
for inputToken in inputTokens:
    # divide by number of tokens to ensure no input node is above 1
    inputTensor[inputToken] += (1 / numOfInputTokens)


# # Download training data from open datasets.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# # Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# batch_size = 64

# Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)

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
inputTensor = inputTensor.to(device)
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

# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")

# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

with torch.no_grad():
    outputTensor = model(inputTensor)

    # Convert it to a prediction
    predictionTensor : torch.Tensor = outputTensor.softmax(dim=0)
    predictionTensor = predictionTensor.cpu()
    predictions = predictionTensor.numpy()
    print(predictions[0])