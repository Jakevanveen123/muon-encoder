import numpy as np
import torch
from config import VOCAB_SIZE, MASK_ID


class TokenStream:
    def __init__(self, path, seq_len, batch_size):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pos = 0

    def next_batch(self):
        n = self.seq_len * self.batch_size
        if self.pos + n > len(self.data):
            self.pos = 0
        chunk = self.data[self.pos : self.pos + n].astype(np.int64)
        self.pos += n
        return torch.from_numpy(chunk).view(self.batch_size, self.seq_len)


def mask_tokens(tokens, mask_rate, generator):
    r = torch.rand(tokens.shape, generator=generator)
    masked = r < mask_rate
    labels = tokens.masked_fill(~masked, -100)
    inputs = tokens.clone()
    inputs[r < 0.8 * mask_rate] = MASK_ID
    random_pos = (r >= 0.8 * mask_rate) & (r < 0.9 * mask_rate)
    random_ids = torch.randint(VOCAB_SIZE, tokens.shape, generator=generator)
    inputs[random_pos] = random_ids[random_pos]
    return inputs, labels
