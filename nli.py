from icecream import ic
import torchtext
from torchtext.vocab import GloVe
from tqdm import tqdm
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import time
import pickle as pkl
from sklearn.metrics import classification_report
import plotly.express as px
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 32
GLOVE_DIM = 200
LEARNING_RATE = 0.001
HIDDEN_SIZE = 200
EPOCHS = 5

datasetMain = load_dataset("multi_nli")
datasetMain = datasetMain.filter(lambda x: x["label"] != -1)

datasetMain = datasetMain.remove_columns(
    ["promptID", "pairID", "genre", 'premise_binary_parse', 'premise_parse', 'hypothesis_binary_parse', 'hypothesis_parse'])
ic(datasetMain)

sets = ["train", "validation", "test"]


def cleanDataset(dataset):
    print("Cleaning dataset...")
    crapSymbols = ["(", ")", ",", ".", ":", ";", "!",
                   "?", "``", "''", "--", "..."]
    for set in sets:
        for i, row in enumerate(tqdm(dataset[set])):
            for symbol in crapSymbols:
                dataset[set][i]["premise"] = row["premise"].replace(
                    symbol, " ")
                dataset[set][i]["hypothesis"] = row["hypothesis"].replace(
                    symbol, " ")

            # extra space removal
            dataset[set][i]["premise"] = " ".join(
                dataset[set][i]["premise"].split())
            dataset[set][i]["hypothesis"] = " ".join(
                dataset[set][i]["hypothesis"].split())
    return dataset


def tokenizeDataset(dataset):
    print("Tokenizing dataset...")
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    for st in sets:
        for i, row in enumerate(tqdm(dataset[st])):
            dataset[st][i]["tokens_p"] = tokenizer(row["premise"])
            dataset[st][i]["tokens_h"] = tokenizer(row["hypothesis"])

    return dataset


def pruneDataset(dataset):
    print("Pruning dataset...")
    newDataset = {"train": [], "validation": [], "test": []}
    for st in sets:
        for i, row in enumerate(tqdm(dataset[st])):
            if len(row["tokens_p"]) <= 50 and len(row["tokens_h"]) <= 50:
                newDataset[st].append(row)

    return newDataset


if os.path.exists("datasetLocal.pkl"):
    datasetMain = pkl.load(open("datasetLocal.pkl", "rb"))
    ic("Loaded datasetLocal from file")
else:
    datasetLocal = {"train": [], "validation": [], "test": []}
    for i, row in enumerate(tqdm(datasetMain["train"], desc="train")):
        if i < np.ceil(len(datasetMain["train"]) * 0.8):
            datasetLocal["train"].append(
                {"premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]})
        else:
            datasetLocal["test"].append(
                {"premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]})
    for i, row in enumerate(tqdm(datasetMain["validation_matched"], desc="validation_matched")):
        if i < np.ceil(len(datasetMain["validation_matched"]) * 0.8):
            datasetLocal["validation"].append(
                {"premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]})
        else:
            datasetLocal["test"].append(
                {"premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]})

    for i, row in enumerate(tqdm(datasetMain["validation_mismatched"], desc="validation_mismatched")):
        if i < np.ceil(len(datasetMain["validation_mismatched"]) * 0.8):
            datasetLocal["validation"].append(
                {"premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]})
        else:
            datasetLocal["test"].append(
                {"premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]})

    datasetMain = cleanDataset(datasetLocal)
    datasetMain = tokenizeDataset(datasetMain)
    datasetMain = pruneDataset(datasetMain)
    pkl.dump(datasetMain, open("datasetLocal.pkl", "wb"))

ic(len(datasetMain["train"]))
ic(len(datasetMain["validation"]))
ic(len(datasetMain["test"]))

ic(datasetMain["train"][:2])
lengths = []
mx = 0
for st in sets:
    for i, row in enumerate(tqdm(datasetMain[st], desc="Calculating max length")):
        mx = max(mx, len(row["tokens_p"]), len(row["tokens_h"]))
        lengths.append(len(row["tokens_p"]))
        lengths.append(len(row["tokens_h"]))
print("Max length: ", mx)
# plotly histogram
fig = px.histogram(x=lengths, nbins=1000)
# html
fig.write_html("histogram.html")

# pad
for st in sets:
    for i, row in enumerate(tqdm(datasetMain[st], desc="Padding "+st)):
        # eos and sos
        datasetMain[st][i]["tokens_p"] = ["<sos>"] + \
            datasetMain[st][i]["tokens_p"] + ["<eos>"]
        datasetMain[st][i]["tokens_p"] = ["<pad>"] * \
            (mx + 2 - len(datasetMain[st][i]["tokens_p"])) + \
            datasetMain[st][i]["tokens_p"]

        datasetMain[st][i]["tokens_h"] = ["<sos>"] + \
            datasetMain[st][i]["tokens_h"] + ["<eos>"]
        datasetMain[st][i]["tokens_h"] = ["<pad>"] * \
            (mx + 2 - len(datasetMain[st][i]["tokens_h"])) + \
            datasetMain[st][i]["tokens_h"]


vocabulary = torchtext.vocab.build_vocab_from_iterator(
    [row["tokens_h"] + row["tokens_p"] for row in datasetMain["train"]], specials=["<unk>", "<eos>", "<sos>", "<pad>"], special_first=True)
# ic(vocabulary.get_itos()[0:10])
print("Vocabulary size: ", len(vocabulary))
glove = GloVe(name="twitter.27B", dim=GLOVE_DIM)
vocabSet = set(vocabulary.get_itos())
gloveSet = set(glove.stoi.keys())
intersections = vocabSet.intersection(gloveSet)
vocabulary = torchtext.vocab.build_vocab_from_iterator([[intersection] for intersection in intersections], specials=[
    "<unk>", "<eos>", "<sos>", "<pad>"], special_first=True)

vocabulary.set_default_index(vocabulary["<unk>"])
# ic(vocabulary.get_itos()[0:30])
print("Vocabulary size after intersection: ", len(vocabulary))


class SSTDataset(Dataset):
    def __init__(self, dataset, vocabulary):
        super().__init__()
        self.dataset = dataset
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (torch.tensor([self.vocabulary[token] for token in self.dataset[idx]["tokens_p"]]).to(device), torch.tensor([self.vocabulary[token] for token in self.dataset[idx]["tokens_h"]]).to(device)), torch.tensor(self.dataset[idx]["label"]).to(device)


class SSTDatasetLM(Dataset):
    def __init__(self, dataset, vocabulary):
        super().__init__()
        self.dataset = dataset
        self.sentences = []
        for row in self.dataset:
            self.sentences.append(row["tokens_p"])
            self.sentences.append(row["tokens_h"])
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor([self.vocabulary[token] for token in self.sentences[idx]]).to(device)


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

        self.classifier = torch.nn.Linear(4 * h, 3)

    def forward(self, x, mode="train"):
        if mode == "train":
            embed = self.embedding(x)
            f1, _ = self.f1(embed)
            b1, _ = self.b1(torch.flip(embed, [1]))
            b1 = torch.flip(b1, [1])

            f2, _ = self.f2(f1)
            b2, _ = self.b2(torch.flip(b1, [1]))
            b2 = torch.flip(b2, [1])

            return self.final(f2), self.final(b2)

        embed1 = self.embedding(x[0])
        embed2 = self.embedding(x[1])
        f1e1, _ = self.f1(embed1)
        b1e1, _ = self.b1(torch.flip(embed1, [1]))
        b1e1 = torch.flip(b1e1, [1])

        f2e1, _ = self.f2(f1e1)
        b2e1, _ = self.b2(torch.flip(b1e1, [1]))
        b2e1 = torch.flip(b2e1, [1])

        f1e2, _ = self.f1(embed2)
        b1e2, _ = self.b1(torch.flip(embed2, [1]))
        b1e2 = torch.flip(b1e2, [1])

        f2e2, _ = self.f2(f1e2)
        b2e2, _ = self.b2(torch.flip(b1e2, [1]))
        b2e2 = torch.flip(b2e2, [1])
        concatHidden1e1 = torch.cat((f1e1, b1e1), dim=2)
        concatHidden2e1 = torch.cat((f2e1, b2e1), dim=2)

        concatHidden1e2 = torch.cat((f1e2, b1e2), dim=2)
        concatHidden2e2 = torch.cat((f2e2, b2e2), dim=2)

        fe1 = self.finalWeights[0][0] * concatHidden1e1 + self.finalWeights[0][1] * \
            concatHidden2e1 + self.finalWeights[0][2] * embed1.repeat(1, 1, 2)
        fe2 = self.finalWeights[0][0] * concatHidden1e2 + self.finalWeights[0][1] * \
            concatHidden2e2 + self.finalWeights[0][2] * embed2.repeat(1, 1, 2)

        fe1 = torch.mean(fe1, dim=1)
        fe2 = torch.mean(fe2, dim=1)

        concat = torch.cat((fe1, fe2), dim=1)
        return self.classifier(concat)


def train(model, trainData, valData):
    totalLoss = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    prevLoss = 999999999
    prevValLoss = 999999999
    for epoch in range(EPOCHS):
        model.train()
        ELoss = 0
        print("Epoch: ", epoch + 1)
        pbar = tqdm(
            dataLoader, desc=f"Pre-Training")
        cur = 0
        for sentence in pbar:
            optimizer.zero_grad()
            seqLen = mx + 2
            f, b = model(sentence, mode="train")
            f = f[:, 1:, :]
            b = b[:, :seqLen - 1, :]
            fTruth = sentence[:, 1:].to(device)
            bTruth = sentence[:, :-1].to(device)

            fLoss = criterion(f.contiguous(
            ).view(-1, f.shape[2]), fTruth.contiguous().view(-1))
            bLoss = criterion(b.contiguous(
            ).view(-1, b.shape[2]), bTruth.contiguous().view(-1))

            loss = fLoss + bLoss
            ELoss += loss.item()
            cur += 1

            pbar.set_description(
                f"Pre-Training | Loss: {ELoss / cur : .10f}")
            loss.backward()
            optimizer.step()

        prevLoss = ELoss

        with torch.no_grad():
            model.eval()
            ELoss_V = 0
            dataLoaderV = DataLoader(
                valData, batch_size=BATCH_SIZE, shuffle=True)
            pbar = tqdm(
                dataLoaderV, desc=f"Validation")
            cur = 0
            for sentence in pbar:
                seqLen = mx + 2
                f, b = model(sentence, mode="train")
                f = f[:, 1:, :]
                b = b[:, :seqLen - 1, :]
                fTruth = sentence[:, 1:].to(device)
                bTruth = sentence[:, :-1].to(device)

                fLoss = criterion(f.contiguous(
                ).view(-1, f.shape[2]), fTruth.contiguous().view(-1))
                bLoss = criterion(b.contiguous(
                ).view(-1, b.shape[2]), bTruth.contiguous().view(-1))

                loss = fLoss + bLoss
                ELoss_V += loss.item()
                cur += 1

                pbar.set_description(
                    f"Validation | Loss: {ELoss_V / cur : .10f}")

            if prevValLoss > ELoss_V:
                torch.save(model, "elmon.pt")

            prevValLoss = ELoss_V


def trainClassification(model, trainData, valData):
    totalLoss = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    prevLoss = 999999999
    prevValLoss = 999999999
    for epoch in range(EPOCHS):
        model.train()
        ELoss = 0
        print("Epoch: ", epoch + 1)
        pbar = tqdm(
            dataLoader, desc=f"Finetuning")
        cur = 0
        for (sentence, label) in pbar:
            optimizer.zero_grad()
            score = model(sentence, mode="classify")

            loss = criterion(score, label)

            loss.backward()

            optimizer.step()

            ELoss += loss.item()
            cur += 1

            pbar.set_description(
                f"Finetuning | Loss: {ELoss / cur : .10f}")

        prevLoss = ELoss

        with torch.no_grad():
            model.eval()
            ELoss_V = 0
            dataLoaderV = DataLoader(
                valData, batch_size=BATCH_SIZE, shuffle=True)
            pbar = tqdm(
                dataLoaderV, desc=f"Validation")
            cur = 0
            for (sentence, label) in pbar:
                seqLen = mx + 2
                score = model(sentence, mode="classify")

                loss = criterion(score, label)
                ELoss_V += loss.item()

                cur += 1

                pbar.set_description(
                    f"Validation | Loss: {ELoss_V / cur : .10f}")

            # if prevValLoss > ELoss_V:
            #     torch.save(model.state_dict(), "elmoFinal.pt")

            prevValLoss = ELoss_V


def testModel(model, testDataset):
    model.eval()
    dataLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)
    trueVals = np.array([])
    predVals = np.array([])
    with torch.no_grad():
        for (sentence, label) in tqdm(dataLoader, desc="Testing"):
            seqLen = mx + 2
            score = model(sentence, mode="classify")
            # exit(0)
            pred = torch.argmax(score, dim=1)
            trueVals = np.append(trueVals, label.cpu().numpy())
            predVals = np.append(predVals, pred.cpu().numpy())

    print(classification_report(trueVals, predVals))


if not os.path.exists("elmon.pt"):
    elmo = ELMo(HIDDEN_SIZE, vocabulary, glove.vectors).to(device)

    train(elmo, SSTDatasetLM(datasetMain["train"], vocabulary), SSTDatasetLM(
        datasetMain["validation"], vocabulary))
    # save entire model not just dict
    torch.save(elmo, "elmon.pt")

elmo = torch.load("elmon.pt")
vocabulary = elmo.vocab

for param in elmo.f1.parameters():
    param.requires_grad = False

for param in elmo.f2.parameters():
    param.requires_grad = False

for param in elmo.b1.parameters():
    param.requires_grad = False

for param in elmo.b2.parameters():
    param.requires_grad = False

if not os.path.exists("elmoFinalnli.pt"):
    trainClassification(elmo, SSTDataset(datasetMain["train"], vocabulary), SSTDataset(
        datasetMain["validation"], vocabulary))
    torch.save(elmo, "elmoFinalnli.pt")
elmo = torch.load("elmoFinalnli.pt")

print("Testing")
testModel(elmo, SSTDataset(datasetMain["test"], vocabulary))

print("Training")
testModel(elmo, SSTDataset(datasetMain["train"], vocabulary))
