from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq

        self.special_tokens = [
            "<PAD>",
            "<SOS>",
            "<EOS>",
            "<UNK>"
        ]

        self.token_to_idx = {}
        self.idx_to_token = {}

    def build_vocab(self, sentences):
        counter = Counter()

        for sentence in sentences:
            counter.update(self._tokenize(sentence))

        vocab = self.special_tokens.copy()

        for token, freq in counter.items():
            if freq >= self.min_freq:
                vocab.append(token)

        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(vocab)
        }

        self.idx_to_token = {
            idx: token
            for token, idx in self.token_to_idx.items()
        }

    def numericalize(self, sentence):
        tokens = self._tokenize(sentence)

        return [
            self.token_to_idx.get(token, self.token_to_idx["<UNK>"])
            for token in tokens
        ]

    def _tokenize(self, sentence):
        if isinstance(sentence, str):
            return sentence.split()

        return list(sentence)
