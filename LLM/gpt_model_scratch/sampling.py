import tiktoken
import re
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

tokenizer = tiktoken.get_encoding("gpt2")

with open("./the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
sample_text = enc_text[:50]
context_size = 7

x = sample_text[:context_size]
y = sample_text[1:context_size + 1]

for i in range(1, context_size):
    context = tokenizer.decode(sample_text[:i])
    desired = tokenizer.decode([sample_text[i]])



example = "hi, i am lucky namikaze from the hidden leave village and the most powerful shinobi alive and to ever ever existed"

preprocess = re.split(r'([.,;''][\/|"]|--|\s)', example)
preprocess = [item.strip() for item in preprocess if item.strip()]

all_words = sorted(list(set(preprocess)))

vocab = {integer: string for string, integer in enumerate(all_words)}


class GPT1DatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunks = token_ids[i:i + max_length]
            target_chunks = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPT1DatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


with open("./the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

torch.manual_seed(123)


dataloader = create_dataloader_v1(raw_text,8,4,4, False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
vocab_size = 50272
output_dim =256
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
token_embedding = token_embedding_layer(inputs)
print("the token embedding is: ",token_embedding.shape)
pos_embedding_layer = torch.nn.Embedding(4,256)
pos_embedding = pos_embedding_layer(torch.arange(4))
print("the pos embedding is: ",pos_embedding.shape)
input_embedding = token_embedding + pos_embedding
print("the input embedding is: ",input_embedding.shape)


