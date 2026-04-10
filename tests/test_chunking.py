from research_graph.chunking import chunk_text


def test_empty_input():
    assert chunk_text('') == []
    assert chunk_text(None) == []
    assert chunk_text('   ') == []


def test_single_paragraph():
    result = chunk_text('Hello world')
    assert result == ['Hello world']


def test_multiple_paragraphs():
    text = 'First paragraph.\n\nSecond paragraph.\n\nThird paragraph.'
    result = chunk_text(text)
    assert len(result) == 1
    assert 'First paragraph.' in result[0]
    assert 'Second paragraph.' in result[0]
    assert 'Third paragraph.' in result[0]


def test_long_paragraph():
    text = 'x' * 2000
    result = chunk_text(text, max_chars=900)
    assert len(result) >= 2
    for chunk in result:
        assert len(chunk) <= 900


def test_custom_max_chars():
    text = 'Hello.\n\nWorld.'
    result = chunk_text(text, max_chars=50)
    assert len(result) == 1
    assert 'Hello.' in result[0]
    assert 'World.' in result[0]
    # When max_chars is too small for combined paragraphs, they split
    result2 = chunk_text(text, max_chars=8)
    assert len(result2) == 2
    assert result2[0] == 'Hello.'
    assert result2[1] == 'World.'
