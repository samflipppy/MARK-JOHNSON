"""
Kalshi API authentication — RSA-PSS request signing.

Each Kalshi API request requires three headers:
  - KALSHI-ACCESS-KEY:       API key ID from the Kalshi dashboard
  - KALSHI-ACCESS-SIGNATURE: RSA-PSS(SHA-256) signature of (timestamp + method + path)
  - KALSHI-ACCESS-TIMESTAMP: Current time in milliseconds

See: https://docs.kalshi.com/getting_started/api_keys
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import config

logger = logging.getLogger("mark_johnson.kalshi_auth")

_private_key: Any = None
_key_loaded = False


def _load_private_key() -> Any:
    """Load the RSA private key from config (PEM string from .env)."""
    global _private_key, _key_loaded
    if _key_loaded:
        return _private_key

    _key_loaded = True
    pem = config.KALSHI_PRIVATE_KEY_PEM
    if not pem:
        logger.info("No KALSHI_PRIVATE_KEY set — API auth disabled")
        return None

    # .env files often mangle newlines; fix common formats
    pem = pem.replace("\\n", "\n").replace("\\r", "")
    if "-----BEGIN" not in pem:
        pem = "-----BEGIN RSA PRIVATE KEY-----\n" + pem
    if "-----END" not in pem:
        pem = pem.rstrip() + "\n-----END RSA PRIVATE KEY-----\n"

    try:
        _private_key = serialization.load_pem_private_key(
            pem.encode("utf-8"),
            password=None,
        )
        logger.info("Kalshi RSA private key loaded successfully")
        return _private_key
    except Exception as exc:
        logger.error("Failed to load Kalshi RSA private key: %s", exc)
        _private_key = None
        return None


def _sign_pss(message: str) -> str:
    """Sign a message with RSA-PSS / SHA-256 and return base64-encoded signature."""
    key = _load_private_key()
    if key is None:
        raise RuntimeError("Kalshi private key not available")

    signature = key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def is_auth_available() -> bool:
    """Return True if API key + private key are configured."""
    return bool(config.KALSHI_API_KEY_ID) and _load_private_key() is not None


def build_auth_headers(method: str, path: str) -> dict[str, str]:
    """
    Build the three Kalshi auth headers for a given request.

    Args:
        method: HTTP method (GET, POST, etc.)
        path:   Request path without query string (e.g. "/trade-api/v2/markets")
    """
    timestamp_ms = str(int(time.time() * 1000))
    # Strip query parameters — Kalshi signs the path only
    path_clean = path.split("?")[0]
    message = timestamp_ms + method.upper() + path_clean
    signature = _sign_pss(message)

    return {
        "KALSHI-ACCESS-KEY": config.KALSHI_API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
    }
