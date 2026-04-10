from datetime import datetime
import hashlib
import re


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r'[^a-z0-9]+', '-', value)
    value = re.sub(r'-+', '-', value).strip('-')
    return value or 'item'


def stable_id(prefix: str, *parts: str) -> str:
    raw = '||'.join(p or '' for p in parts)
    digest = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]
    return f'{prefix}-{digest}'
