from pathlib import Path

__all__ = ["settings"]


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"


settings = Settings()
