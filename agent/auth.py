"""Mint the bearer token the agent sends to the MCP servers."""
from __future__ import annotations

from fastmcp.server.auth.providers.jwt import RSAKeyPair

from agent.config import settings


def mint_agent_token() -> str | None:
    if not (settings.auth_private_key and settings.auth_public_key):
        return None
    kp = RSAKeyPair(public_key=settings.auth_public_key, private_key=settings.auth_private_key)
    return kp.create_token(
        subject="signal-mcp-agent",
        issuer=settings.auth_issuer,
        audience=settings.auth_audience,
        scopes=["mcp.call"],
        expires_in_seconds=3600,
    )
