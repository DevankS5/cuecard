"""Run with `python test_cuecard.py` (or pytest). No mic, model or server needed."""
from cuecard import parse_chunks


def test_parse_chunks_joins_stream_and_stops_at_done():
    lines = [
        b'{"message":{"content":"I "},"done":false}\n',
        b"\n",
        b'{"message":{"content":"led it."},"done":false}\n',
        b'{"message":{"content":""},"done":true}\n',
        b'{"message":{"content":"ignored"},"done":false}\n',
    ]
    assert "".join(parse_chunks(lines)) == "I led it."


def test_parse_chunks_raises_server_error():
    try:
        list(parse_chunks([b'{"error":"model \\"x\\" not found"}\n']))
    except RuntimeError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


if __name__ == "__main__":
    test_parse_chunks_joins_stream_and_stops_at_done()
    test_parse_chunks_raises_server_error()
    print("ok")
