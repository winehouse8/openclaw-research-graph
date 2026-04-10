from __future__ import annotations

from typing import List
import json
import os
import urllib.request


OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/embed')
EMBED_MODEL = os.environ.get('RESEARCH_GRAPH_EMBED_MODEL', 'qwen3-embedding:0.6b')


def fake_embed(text: str, dims: int = 1024) -> List[float]:
    text = (text or '').strip()
    if not text:
        return [0.0] * dims
    base = [0.0] * dims
    for i, ch in enumerate(text.encode('utf-8')):
        base[i % dims] += (ch % 31) / 31.0
    norm = sum(x * x for x in base) ** 0.5 or 1.0
    return [x / norm for x in base]


def embed_text(text: str, model: str = EMBED_MODEL) -> List[float]:
    text = (text or '').strip()
    if not text:
        return [0.0] * 1024
    payload = json.dumps({'model': model, 'input': text}).encode('utf-8')
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        embeddings = data.get('embeddings') or []
        if not embeddings:
            return fake_embed(text)
        return embeddings[0]
    except (urllib.error.URLError, OSError):
        return fake_embed(text)
