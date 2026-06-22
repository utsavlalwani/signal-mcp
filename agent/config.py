"""Shared settings for all MCP servers and the agent. Single source of truth for env-driven config."""
from __future__ import annotations
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # Upstream data APIs
    fred_api_key: str = ""
    edgar_user_agent: str = "Example User example@example.com"
    polygon_api_key: str = ""
    tiingo_api_key: str = ""
    market_data_provider: Literal["yfinance", "polygon", "tiingo"] = "yfinance"

    # LLM
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    agent_llm_provider: Literal["groq", "anthropic"] = "groq"
    agent_model: str = "llama-3.3-70b-versatile"

    # Observability
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    otel_exporter_otlp_endpoint: str = "http://localhost:3001/api/public/otel"
    otel_service_name: str = "signal-mcp"

    # Auth (OAuth 2.1 bearer, RS256 JWT) -- keys minted by scripts/mint_dev_token.py
    auth_public_key: str = ""
    auth_private_key: str = ""
    auth_issuer: str = "https://signal-mcp.local"
    auth_audience: str = "signal-mcp"

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "signal"
    postgres_password: str = "signal_dev_password"
    postgres_db: str = "signalmcp"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # MCP ports
    market_data_port: int = 8001
    edgar_fundamentals_port: int = 8002
    fred_macro_port: int = 8003
    quant_engine_port: int = 8004
    mcp_base_url: str = "http://localhost"

    @property
    def server_urls(self) -> dict[str, str]:
        return {
            "market_data": f"{self.mcp_base_url}:{self.market_data_port}/mcp",
            "edgar_fundamentals": f"{self.mcp_base_url}:{self.edgar_fundamentals_port}/mcp",
            "fred_macro": f"{self.mcp_base_url}:{self.fred_macro_port}/mcp",
            "quant_engine": f"{self.mcp_base_url}:{self.quant_engine_port}/mcp",
        }


settings = Settings()
