"""Keeps API keys out of child processes and out of model context."""

import os
import re

_SECRET_NAME = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)


def is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME.search(name))


def scrubbed_env() -> dict[str, str]:
    """The current environment minus anything that looks like a credential."""
    return {k: v for k, v in os.environ.items() if not is_secret_name(k)}


def redact(text: str) -> str:
    """Replace the value of any secret-looking environment variable found in `text`."""
    for name, value in os.environ.items():
        if is_secret_name(name) and len(value) >= 8 and value in text:
            text = text.replace(value, f"[redacted ${name}]")
    return text
