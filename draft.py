from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch

# 1. List captions
captions = [
    "Two young, White males are outside near many bushes.",
    "Several men in hard hats are operating a giant pulley system.",
    "A little girl climbing into a wooden playhouse.",
    "A man in a blue shirt is standing on a ladder cleaning a window.",
    "Two men are at the stove preparing food."
]

# 2. Tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# 3. Tokenize từng câu
tokenized = [tokenizer(c) for c in captions]

# 4. Tạo vocab tạm thời (token -> index)
all_tokens = [tok for sent in tokenized for tok in sent]
vocab = {tok: idx+1 for idx, tok in enumerate(sorted(set(all_tokens)))}  # +1 để reserve 0 cho <pad>
pad_idx = 0

# 5. Chuyển token -> index
batch_indices = [torch.tensor([vocab[tok] for tok in sent]) for sent in tokenized]

# 6. Tính lengths
lengths = torch.tensor([len(seq) for seq in batch_indices])

# 7. Pad
src_padded = pad_sequence(batch_indices, batch_first=True, padding_value=pad_idx)

# 8. Sắp xếp giảm dần
lengths_sorted, sorted_idx = torch.sort(lengths, descending=True)
src_sorted = src_padded[sorted_idx]

# 9. Pack
packed_input = pack_padded_sequence(src_sorted, lengths_sorted, batch_first=True, enforce_sorted=True)

# --- DEBUG: in thông tin từng câu ---
for i, seq in enumerate(batch_indices):
    print(f"Original sentence: {captions[i]}")
    print(f"Tokenized: {tokenized[i]}")
    print(f"Length (number of tokens): {len(seq)}")
    print('-'*50)

print("Lengths sorted (for PackedSequence):", lengths_sorted)
print("Packed batch data shape:", packed_input.data.shape)
