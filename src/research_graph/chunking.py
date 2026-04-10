from typing import List


def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    text = (text or '').strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    buf = ''
    for para in paras:
        if len(buf) + len(para) + 2 <= max_chars:
            buf = f'{buf}\n\n{para}'.strip()
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= max_chars:
                buf = para
            else:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars])
                buf = ''
    if buf:
        chunks.append(buf)
    return chunks
