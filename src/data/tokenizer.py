from pathlib import Path


class SentencePieceTokenizer:
    def __init__(self, model_path):
        sentencepiece = _import_sentencepiece()

        self.model_path = str(model_path)
        self.processor = sentencepiece.SentencePieceProcessor()
        self.processor.load(self.model_path)

        self.token_to_idx = {
            self.processor.id_to_piece(idx): idx
            for idx in range(self.processor.get_piece_size())
        }
        self.idx_to_token = {
            idx: piece
            for piece, idx in self.token_to_idx.items()
        }

    def __len__(self):
        return self.processor.get_piece_size()

    @property
    def pad_id(self):
        return self.processor.pad_id()

    @property
    def sos_id(self):
        return self.processor.bos_id()

    @property
    def eos_id(self):
        return self.processor.eos_id()

    @property
    def unk_id(self):
        return self.processor.unk_id()

    def numericalize(self, sentence):
        if isinstance(sentence, str):
            return self.processor.encode(sentence, out_type=int)

        return [
            self.token_to_idx.get(token, self.unk_id)
            for token in sentence
        ]

    def decode(self, token_ids):
        special_ids = {
            self.pad_id,
            self.sos_id,
            self.eos_id,
        }
        token_ids = [
            token_id
            for token_id in token_ids
            if token_id not in special_ids
        ]

        return self.processor.decode(token_ids)


def train_sentencepiece_tokenizer(
    sentences,
    model_prefix,
    vocab_size,
    model_type="bpe"
):
    sentencepiece = _import_sentencepiece()

    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    corpus_path = model_prefix.with_suffix(".txt")
    corpus_path.write_text(
        "\n".join(sentences),
        encoding="utf-8"
    )

    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<PAD>",
        bos_piece="<SOS>",
        eos_piece="<EOS>",
        unk_piece="<UNK>",
        hard_vocab_limit=False,
    )

    return SentencePieceTokenizer(
        model_prefix.with_suffix(".model")
    )


def _import_sentencepiece():
    try:
        import sentencepiece
    except ImportError as exc:
        raise ImportError(
            "sentencepiece is required for SentencePieceTokenizer. "
            "Install it with `pip install sentencepiece`."
        ) from exc

    return sentencepiece
