# GPT-2 model

## Installation
the model can be run both as `.ipynb` or by running `main.py`

## Dataset
We're using the `tiny shakespeare` dataset to evaulate the model on over 1M keys.
The Model will be able print a combination of `65` chars.

## Data Preprosessing
Data we have is strings and we want to convert it to numarics inorder to feed it to the model. The `get_batch` method
```py
# src/utils
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

def get_batch(split, data):
```
## Architecture
We're using a GPT-2 based model, it predicts subsequent charecters based on a given sequence.

```py
class Transformer(nn.Module):
```
