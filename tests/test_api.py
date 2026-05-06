import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.mark.asyncio
async def test_auth_flow():
    # Naye httpx version ke mutabiq ASGITransport define karna zaroori hai
    transport = ASGITransport(app=app)
    
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Register
        reg_res = await ac.post(
            "/api/v1/auth/register", 
            json={"username": "testuser", "email": "test@test.com", "password": "pass"}
        )
        assert reg_res.status_code in [201, 400] # 400 if already exists

        # 2. Login
        login_res = await ac.post(
            "/api/v1/auth/token", 
            data={"username": "testuser", "password": "pass"}
        )
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]

        # 3. Query without JWT (Should fail 401)
        q_fail = await ac.post("/api/v1/query", json={"query": "test"})
        assert q_fail.status_code == 401

        # 4. Documents list API Test with JWT
        headers = {"Authorization": f"Bearer {token}"}
        docs_res = await ac.get("/api/v1/documents/", headers=headers)
        assert docs_res.status_code == 200
        assert "items" in docs_res.json()