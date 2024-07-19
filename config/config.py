from dotenv import find_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_token: str = Field(
        alias='hf_token',
        validation_alias='HF_TOKEN',
    )

    diarization_pipeline_name: str = Field(
        default='pyannote/speaker-diarization@2.1',
        alias='diarization_pipeline_name',
        validation_alias='DIARIZATION_PIPELINE_NAME',
    )

    pytorch_enable_mps_fallback: str = Field(
        default='1',
        alias='pytorch_enable_mps_fallback',
        validation_alias='PYTORCH_ENABLE_MPS_FALLBACK',
    )

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding='utf-8',
        case_sensitive=False,
    )


settings = Settings()