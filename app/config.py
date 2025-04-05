from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "Virtual Try-On API"
    API_VERSION: str = "1.0.0"
    
    # Path Settings
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    VITON_DIR: Path = BASE_DIR / "VITON-HD"
    CHECKPOINT_DIR: Path = VITON_DIR / "checkpoints"
    TEMP_DIR: Path = BASE_DIR / "temp"
    
    # File Settings
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png"}
    
    # Model Settings
    USE_CUDA: bool = True

    class Config:
        env_file = ".env"

settings = Settings()