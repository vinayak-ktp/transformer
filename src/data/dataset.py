import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_sentences,
        tgt_sentences,
        src_vocab,
        tgt_vocab
    ):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]

        src_tokens = self.src_vocab.numericalize(src_sentence)

        tgt_tokens = [
            self.tgt_vocab.token_to_idx["<SOS>"]
        ]

        tgt_tokens += self.tgt_vocab.numericalize(tgt_sentence)

        tgt_tokens += [
            self.tgt_vocab.token_to_idx["<EOS>"]
        ]

        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_tokens, dtype=torch.long)
        }
