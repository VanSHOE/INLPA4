from icecream import ic
import torchtext
from torchtext.vocab import GloVe
from tqdm import tqdm
from datasets import load_dataset

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
