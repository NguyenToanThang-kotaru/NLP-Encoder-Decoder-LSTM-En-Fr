#generating vocab from text file
import warnings
warnings.filterwarnings("ignore", message=".*TORCHTEXT STATUS.*")

import torchtext
torchtext.disable_torchtext_deprecation_warning()

import io
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
file_path = './Data/train.en'

def yield_tokens(file_path):
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            yield en_tokenizer(line.strip())

vocab = build_vocab_from_iterator(yield_tokens(file_path), specials=["<unk>"])
for token in vocab.get_itos():
    print(token)
