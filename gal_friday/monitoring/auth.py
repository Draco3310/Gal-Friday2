"""Authentication and authorization for monitoring dashboard."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gal_friday.config_manager import ConfigManager

security = HTTPBearer()


def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    config: Annotated[ConfigManager, Depends(lambda: ConfigManager())],
) -> str:
    """Verify API key for dashboard access."""
    expected_token = config.get("dashboard.api_token", "dev-token")

    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials
