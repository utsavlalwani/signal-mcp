"""Generate a dev RSA keypair + sample token. Paste the PEMs into .env
(AUTH_PUBLIC_KEY / AUTH_PRIVATE_KEY). Run once."""
from fastmcp.server.auth.providers.jwt import RSAKeyPair

kp = RSAKeyPair.generate()
print("AUTH_PUBLIC_KEY (single line, replace newlines with \\n if needed):\n")
print(kp.public_key)
print("\nAUTH_PRIVATE_KEY:\n")
print(kp.private_key.get_secret_value())
print("\nSample token:\n")
print(kp.create_token(subject="dev", issuer="https://signal-mcp.local",
                      audience="signal-mcp", scopes=["mcp.call"]))
