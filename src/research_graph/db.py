from contextlib import contextmanager
from neo4j import GraphDatabase
from .config import get_settings


@contextmanager
def session_scope():
    s = get_settings()
    driver = GraphDatabase.driver(s.uri, auth=(s.user, s.password))
    try:
        with driver.session() as session:
            yield session
    finally:
        driver.close()
