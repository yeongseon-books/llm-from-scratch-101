from common import CharTokenizer


def test_tokenizer_roundtrip() -> None:
    tok = CharTokenizer("abc xyz")
    ids = tok.encode("cab")
    assert tok.decode(ids) == "cab"
