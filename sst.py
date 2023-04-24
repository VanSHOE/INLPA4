from icecream import ic
import torchtext
from torchtext.vocab import GloVe
from tqdm import tqdm
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, ConfusionMatrixDisplay
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
GLOVE_DIM = 200
LEARNING_RATE = 0.001
HIDDEN_SIZE = 200
EPOCHS = 50
PATIENCE = 5

datasetMain = load_dataset("sst")
datasetMain = datasetMain.remove_columns(["tokens", "tree"])
sets = ["train", "validation", "test"]
datasetLocal = {"train": [], "validation": [], "test": []}
for st in sets:
    for i, row in enumerate(tqdm(datasetMain[st])):
        datasetLocal[st].append(
            {"sentence": row["sentence"], "label": np.round(row["label"]).astype(int)})

ic(datasetLocal["train"][0:10])


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
    for st in sets:
        for i, row in enumerate(tqdm(dataset[st])):
            dataset[st][i]["tokens"] = tokenizer(row["sentence"])

    return dataset


datasetMain = cleanDataset(datasetLocal)
datasetMain = tokenizeDataset(datasetMain)


mx = 0
for st in sets:
    for i, row in enumerate(tqdm(datasetMain[st], desc="Calculating max length")):
        mx = max(mx, len(row["tokens"]))
print("Max length: ", mx)
# pad
for st in sets:
    for i, row in enumerate(tqdm(datasetMain[st], desc="Padding "+st)):
        # eos and sos
        datasetMain[st][i]["tokens"] = ["<sos>"] + \
            datasetMain[st][i]["tokens"] + ["<eos>"]
        datasetMain[st][i]["tokens"] = ["<pad>"] * \
            (mx + 2 - len(datasetMain[st][i]["tokens"])) + \
            datasetMain[st][i]["tokens"]

vocabulary = torchtext.vocab.build_vocab_from_iterator(
    [row["tokens"] for row in datasetMain["train"]], specials=["<unk>", "<eos>", "<sos>", "<pad>"], special_first=True)
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
        return torch.tensor([self.vocabulary[token] for token in self.dataset[idx]["tokens"]]).to(device), torch.tensor(self.dataset[idx]["label"]).to(device)


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

        self.classifier = torch.nn.Linear(2 * h, 2)

    def forward(self, x, mode="train"):
        embed = self.embedding(x)
        f1, _ = self.f1(embed)
        b1, _ = self.b1(torch.flip(embed, [1]))
        b1 = torch.flip(b1, [1])

        f2, _ = self.f2(f1)
        b2, _ = self.b2(torch.flip(b1, [1]))
        b2 = torch.flip(b2, [1])
        if mode == "train":
            return self.final(f2), self.final(b2)

        concatHidden1 = torch.cat((f1, b1), dim=2)
        concatHidden2 = torch.cat((f2, b2), dim=2)

        return self.finalWeights[0][0] * concatHidden1 + self.finalWeights[0][1] * concatHidden2 + self.finalWeights[0][2] * embed.repeat(1, 1, 2)


def train(model, trainData, valData):
    totalLoss = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    prevLoss = 999999999
    prevValLoss = 999999999
    curPat = PATIENCE
    for epoch in range(EPOCHS):
        model.train()
        ELoss = 0
        print("Epoch: ", epoch + 1)
        pbar = tqdm(
            dataLoader, desc=f"Pre-Training")
        cur = 0
        for (sentence, label) in pbar:
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
            for (sentence, label) in pbar:
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
                torch.save(model, "elmo.pt")
                curPat = PATIENCE
            else:
                curPat -= 1
                if curPat == 0:
                    break

            prevValLoss = ELoss_V


def trainClassification(model, trainData, valData):
    totalLoss = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    prevLoss = 999999999
    prevValLoss = 999999999
    curPat = PATIENCE
    for epoch in range(EPOCHS):
        model.train()
        ELoss = 0
        print("Epoch: ", epoch + 1)
        pbar = tqdm(
            dataLoader, desc=f"Finetuning")
        cur = 0
        for (sentence, label) in pbar:
            optimizer.zero_grad()

            seqLen = mx + 2
            state = torch.sum(model(sentence, mode="classify"), dim=1)
            score = model.classifier(state)

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
                state = torch.sum(model(sentence, mode="classify"), dim=1)
                score = model.classifier(state).squeeze()

                loss = criterion(score, label)
                ELoss_V += loss.item()

                cur += 1

                pbar.set_description(
                    f"Validation | Loss: {ELoss_V / cur : .10f}")

            if prevValLoss > ELoss_V:
                torch.save(model, "elmoFinal_sst.pt")
                curPat = PATIENCE
            else:
                curPat -= 1
                if curPat == 0:
                    break

            prevValLoss = ELoss_V


def testModel(model, testDataset, test=True):
    model.eval()
    dataLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)
    trueVals = np.array([])
    predVals = np.array([])
    predValProbs = np.array([])
    with torch.no_grad():
        for (sentence, label) in tqdm(dataLoader, desc="Testing"):
            seqLen = mx + 2
            state = torch.sum(model(sentence, mode="classify"), dim=1)
            score = model.classifier(state).squeeze()
            # ic(score.shape, label.shape)
            # exit(0)

            probs = torch.softmax(score, dim=1)[:, 1]

            pred = torch.argmax(score, dim=1)
            trueVals = np.append(trueVals, label.cpu().numpy())
            predVals = np.append(predVals, pred.cpu().numpy())
            predValProbs = np.append(predValProbs, probs.cpu().numpy())

    print(classification_report(trueVals, predVals))
    if test:
        confusion = confusion_matrix(trueVals, predVals)
        cm = ConfusionMatrixDisplay(confusion)
        cm.plot()
        # save
        plt.savefig("confusionSST.png")
        # clear plt
        plt.clf()
        # ic(predValProbs.shape, trueVals.shape)
        roc = roc_curve(trueVals, predValProbs, pos_label=1)
        # axis names
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot(roc[0], roc[1])

        plt.savefig("rocSST.png")


if not os.path.exists("elmo.pt"):
    elmo = ELMo(HIDDEN_SIZE, vocabulary, glove.vectors).to(device)
    train(elmo, SSTDataset(datasetMain["train"], vocabulary), SSTDataset(
        datasetMain["validation"], vocabulary))


elmo = torch.load("elmo.pt")
vocabulary = elmo.vocab

for param in elmo.f1.parameters():
    param.requires_grad = False

for param in elmo.f2.parameters():
    param.requires_grad = False

for param in elmo.b1.parameters():
    param.requires_grad = False

for param in elmo.b2.parameters():
    param.requires_grad = False

if not os.path.exists("elmoFinal_sst.pt"):
    trainClassification(elmo, SSTDataset(datasetMain["train"], vocabulary), SSTDataset(
        datasetMain["validation"], vocabulary))

elmo = torch.load("elmoFinal_sst.pt")

print("Testing")
testModel(elmo, SSTDataset(datasetMain["test"], vocabulary))

print("Training")
testModel(elmo, SSTDataset(datasetMain["train"], vocabulary), test=False)
