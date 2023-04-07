import json

from model import NeuralNetwork

from nltk_tools import tokenizer, stemmer, compareWords
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

gAllwords = []

def get_result(input):
    return tags[torch.argmax(gTrained(torch.tensor(compareWords(input, gAllwords))))]

def train():
    global gTrained,tags
    tags = set()
    xy = []

    with open('intents.json','r') as f:
        intents = json.load(f)

    for intent in intents['intents']:
        tag=intent['tag']
        tags.add(tag)
        for pattern in intent['patterns']:
            ignore = ["?", "!", ",", "."]
            w = [stemmer(x) for x in tokenizer(pattern) if x not in ignore]
            gAllwords.extend(w)
            xy.append((w,tag))
    print(gAllwords)
    tags = sorted(tags)
    print(tags)
    print(xy)

    X_train = []
    Y_train = []

    for (pattern_sentence,tag) in xy:
        bag = compareWords(pattern_sentence, gAllwords)
        X_train.append(bag)

        label = tags.index(tag)
        Y_train.append(label)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = Y_train

        def __getitem__(self,index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    dataset = ChatDataset()

    #parametry
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    batch_size = 8
    learning_rate = 0.0001
    num_epochs = 5000
    #print(input_size, len(gAllwords))
    print("X_TRAIN: ",X_train)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size)

    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
    #optymalizacja i loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #trenowanie modelu
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)


            outputs = model(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()


            loss.backward()
            optimizer.step()
        if ((epoch+1) %100) == 0:
            print("Loss: ",loss.item(), " epoch:", (epoch+1)/num_epochs)
    print("final loss: ", loss.item())
    gTrained = model

    data = {
        #dane modelu do zapisania
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": gAllwords,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data,FILE)

    print(f'trainging done, saved to {FILE}')