import pytest
from fastapi.testclient import TestClient
from api.index import app
import os
import shutil
import tempfile

@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        os.environ["TESTING"] = "True"
        with TestClient(app) as test_client:
            yield test_client

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)