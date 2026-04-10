from dataclasses import dataclass
import os


@dataclass
class Settings:
    uri: str = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    user: str = os.environ.get('NEO4J_USER', 'neo4j')
    password: str = os.environ.get('NEO4J_PASSWORD', '')


def get_settings() -> Settings:
    return Settings()
