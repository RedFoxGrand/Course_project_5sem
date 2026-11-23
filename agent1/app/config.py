from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    llm_api_key: str = "mock-key"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0
    use_mock_llm: bool = True
    rules_path: Path = BASE_DIR / "data" / "kb" / "requirements_rules.yaml"
    projects_dir: Path = BASE_DIR / "data" / "projects"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
