import pytest
from httpx import AsyncClient, ASGITransport

from api.main import app

@pytest.mark.asyncio
async def test_health_check_returns_200():
    """Validates that the base health endpoint operates without dependencies."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

@pytest.mark.asyncio
async def test_health_ready_returns_200_when_services_up():
    """
    Validates readiness check with backing services.
    Requires Postgres and Redis to be running on localhost (handled by CI services).
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Trigger startup event manually for testing lifespan state
        async with app.router.lifespan_context(app):
            response = await client.get("/health/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["services"]["postgres"] == "up"
            assert data["services"]["redis"] == "up"