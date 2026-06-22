"""Bearer-token auth shared by all MCP servers.

Each server is an OAuth 2.1 resource server: it validates RS256-signed JWT
bearer tokens (issuer / audience / expiry / scopes) using a public key.
Tokens are minted by the agent/client with the matching private key.
If no public key is configured, auth is disabled (local dev convenience).
"""
from __future__ import annotations

from fastmcp.server.auth.providers.jwt import JWTVerifier

from agent.config import settings


def build_verifier() -> JWTVerifier | None:
    if not settings.auth_public_key:
        return None
    return JWTVerifier(
        public_key=settings.auth_public_key,
        issuer=settings.auth_issuer,
        audience=settings.auth_audience,
        algorithm="RS256",
        required_scopes=["mcp.call"],
    )
