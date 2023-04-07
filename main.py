from nltk_tools import *
import train

#train
train.train()
#---

def main():
    while True:
        s = input("Me:")
        s = [stemmer(x) for x in tokenizer(s)]
        print(train.get_result(s))

#main()
