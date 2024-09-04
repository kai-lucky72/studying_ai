import re

# Read and preprocess the text from the file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
example = "hi, i am lucky namikaze from the hidden leave village and the most powerful shinobi alive and to ever ever existed"
preprocessed = re.split(r'([,.?_!;():\']|--|\s)', example)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


# Create the initial vocabulary
all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)

# Create the vocabulary dictionary
vocab = {token: integer for integer, token in enumerate(all_words)}



# Define the tokenizer class
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"():\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"():\'])', r'\1', text)
        return text

# Instantiate the tokenizer
tokenizer = SimpleTokenizerV1(vocab)
print(tokenizer.encode(example))


# Extend the token list with two empty strings and recreate the vocabulary
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}




# Define the new tokenizer class
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"():\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"():\'])', r'\1', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

tokenizer = SimpleTokenizerV2(vocab)
encoded_text = tokenizer.encode(text)
decoded_text = tokenizer.decode(encoded_text)

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "umuntu yagiye kugura imbeba kwisoko ahura na lucky"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
string = tokenizer.decode(integers)
# print(f"GPT-2 Encoded: {integers}")
# print(f"GPT-2 Decoded: {string}")


