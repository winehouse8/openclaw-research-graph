from research_graph.utils import now_iso, slugify, stable_id


def test_now_iso():
    result = now_iso()
    assert 'T' in result
    assert '+' in result or '-' in result[10:]


def test_slugify_spaces():
    assert slugify('Hello World') == 'hello-world'


def test_slugify_special_chars():
    assert slugify('foo@bar!baz') == 'foo-bar-baz'


def test_slugify_empty():
    assert slugify('') == 'item'
    assert slugify('   ') == 'item'


def test_slugify_already_clean():
    assert slugify('hello') == 'hello'


def test_slugify_consecutive_special():
    assert slugify('a---b___c') == 'a-b-c'


def test_stable_id_deterministic():
    id1 = stable_id('prefix', 'a', 'b')
    id2 = stable_id('prefix', 'a', 'b')
    assert id1 == id2


def test_stable_id_prefix():
    result = stable_id('obj', 'x')
    assert result.startswith('obj-')


def test_stable_id_different_inputs():
    id1 = stable_id('p', 'a', 'b')
    id2 = stable_id('p', 'c', 'd')
    assert id1 != id2
