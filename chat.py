import torch,json,random
from model import NeuralNetwork
from nltk_tools import compareWords, tokenizer,stemmer
FILE = "data.pth"


def openModel():

    #sprawdzenie czy jest obsługa gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('intents.json','r') as json_data:
        intents = json.load(json_data)
    data = torch.load(FILE)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data['tags']
    model_state = data['model_state']

    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval() #tryb ewaluacji

    bot_name = "Adrian"
    print("Hello garbage, type 'quit' to quit")
    #bot czeka na nasz wpis
    while True:
        s = input("You:")
        if s=="quit": break
        s = [stemmer(x) for x in tokenizer(s)]
        s = compareWords(s, all_words)
        s = s.reshape(1, s.shape[0])
        s = torch.from_numpy(s).to(device)
        out = model(s)
        _,predicted = torch.max(out, dim=1)
        tag = tags[predicted.item()]
        #print(_)
        #minimalne prawdopodobienstwo do odpowiedzi
        probs = torch.softmax(out, dim=1)
        prob = probs[0][predicted]
        if prob.item()>0.7:
            for intent in intents['intents']:
                if tag==intent['tag']:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
                    break
            else:
                print(f'{bot_name}: co')
        else:
            print(f'{bot_name}: co', prob.item())
openModel()
