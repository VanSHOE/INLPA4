from icecream import ic
import torchtext
from torchtext.vocab import GloVe
from tqdm import tqdm
from datasets import load_dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOVE_DIM = 200

datasetMain = load_dataset("sst")
datasetMain = datasetMain.remove_columns(["tokens", "tree"])
sets = ["train", "validation", "test"]
datasetLocal = {"train": [], "validation": [], "test": []}
for st in sets:
    for i, row in enumerate(tqdm(datasetMain[st])):
        datasetLocal[st].append(
            {"sentence": row["sentence"], "label": row["label"]})


def cleanDataset(dataset):
    print("Cleaning dataset...")
    crapSymbols = ["(", ")", ",", ".", ":", ";", "!",
                   "?", "``", "''", "--", "..."]
    for set in sets:
        for i, row in enumerate(tqdm(dataset[set])):
            for symbol in crapSymbols:
                dataset[set][i]["sentence"] = row["sentence"].replace(
                    symbol, " ")

            # extra space removal
            dataset[set][i]["sentence"] = " ".join(
                dataset[set][i]["sentence"].split())
    return dataset


def tokenizeDataset(dataset):
    print("Tokenizing dataset...")
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    for set in sets:
        for i, row in enumerate(tqdm(dataset[set])):
            dataset[set][i]["tokens"] = tokenizer(row["sentence"])

    return dataset


datasetMain = cleanDataset(datasetLocal)
datasetMain = tokenizeDataset(datasetMain)
vocabulary = torchtext.vocab.build_vocab_from_iterator(
    [row["tokens"] for row in datasetMain["train"]], specials=["<unk>", "<eos>", "<sos>", "<pad>"], special_first=True)

ic(vocabulary.get_itos()[0:10])
print("Vocabulary size: ", len(vocabulary))
glove = GloVe(name="twitter.27B", dim=GLOVE_DIM)
vocabSet = set(vocabulary.get_itos())
gloveSet = set(glove.stoi.keys())
intersections = vocabSet.intersection(gloveSet)
vocabulary = torchtext.vocab.build_vocab_from_iterator([[intersection] for intersection in intersections], specials=[
                                                       "<unk>", "<eos>", "<sos>", "<pad>"], special_first=True)
ic(vocabulary.get_itos()[0:30])
print("Vocabulary size: ", len(vocabulary))


class ELMo(torch.nn.Module):
    def __init__(self, h, vocab, glEmbed):
        super().__init__()
        self.vocab = vocab
        self.glove_dim = GLOVE_DIM
        self.hidden_size = h
        self.vocab_size = len(vocab)
        self.embedding = torch.nn.Embedding.from_pretrained(glEmbed)
        self.f1 = torch.nn.LSTM(self.glove_dim, h, batch_first=True)
        self.b1 = torch.nn.LSTM(self.glove_dim, h, batch_first=True)

        self.f2 = torch.nn.LSTM(h, h, batch_first=True)
        self.b2 = torch.nn.LSTM(h, h, batch_first=True)

        self.final = torch.nn.Linear(h, self.vocab_size)
        self.finalWeights = torch.nn.Parameter(torch.randn((1, 3)).to(device))

    def forward(self, x, mode="train"):
        x = self.embedding(x)

        f1, _ = self.f1(x)
        b1, _ = self.b1(torch.flip(x, [1]))
        b1 = torch.flip(b1, [1])

        f2, _ = self.f2(f1)
        b2, _ = self.b2(torch.flip(b1, [1]))
        b2 = torch.flip(b2, [1])

        if mode == "train":
            return self.final(f2), self.final(b2)
