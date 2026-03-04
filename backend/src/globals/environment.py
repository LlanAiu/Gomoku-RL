# builtin

# external
from pydantic_settings import BaseSettings, SettingsConfigDict

# internal


class Environment(BaseSettings):
    ENV_TYPE: str
    ALLOWED_ORIGIN: str
    
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=(".env.development", ".env.production", ".env")
    )