from pathlib import Path
import subprocess
import sys
import textwrap
from tempfile import TemporaryDirectory



def test_sentencepiece_tokenizer_trains_and_round_trips_text():
    with TemporaryDirectory() as tmp_dir:
        script = textwrap.dedent(
            f"""
            from pathlib import Path

            from src.data.tokenizer import train_sentencepiece_tokenizer

            tokenizer = train_sentencepiece_tokenizer(
                ["i love ai", "hello world", "thank you"],
                model_prefix=Path({str(Path(tmp_dir) / "toy")!r}),
                vocab_size=32,
            )
            token_ids = tokenizer.numericalize("hello ai")

            assert len(tokenizer) > 0
            assert tokenizer.pad_id == 0
            assert tokenizer.sos_id == 1
            assert tokenizer.eos_id == 2
            assert tokenizer.unk_id == 3
            assert tokenizer.decode(token_ids) == "hello ai"
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
