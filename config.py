from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class ABTestingChatbotSettings(BaseSettings):
    """Configuración del chatbot de AB Testing"""
    
    # OpenAI Configuration
    openai_api_key: SecretStr
    openai_model_id: str = "gpt-4"
    
    # Dataset Configuration
    dataset_path: str = "tiendas_detalle.csv"
    
    # Superlinked Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Query Configuration
    default_limit: int = 5
    similarity_threshold: float = 0.7
    
    # Espacios vectoriales - pesos configurables
    title_space_weight: float = 1.0
    description_space_weight: float = 1.0
    region_space_weight: float = 0.8
    tipo_tienda_space_weight: float = 0.8
    experimento_space_weight: float = 0.9
    usuarios_space_weight: float = 0.6
    conversiones_space_weight: float = 0.8
    revenue_space_weight: float = 0.8
    conversion_rate_space_weight: float = 0.9
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }
    
    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Dataset file not found: {v}")
        return v

# Instancia global de configuración
settings = ABTestingChatbotSettings()