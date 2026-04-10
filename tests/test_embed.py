import math

from research_graph.embed import fake_embed


def test_fake_embed_dimensions():
    result = fake_embed('hello world')
    assert len(result) == 1024


def test_fake_embed_empty():
    result = fake_embed('')
    assert len(result) == 1024
    assert all(x == 0.0 for x in result)


def test_fake_embed_deterministic():
    a = fake_embed('test input')
    b = fake_embed('test input')
    assert a == b


def test_fake_embed_normalized():
    result = fake_embed('some text here')
    norm = math.sqrt(sum(x * x for x in result))
    assert abs(norm - 1.0) < 1e-6
