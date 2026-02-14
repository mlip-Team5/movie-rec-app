import pytest

from app import create_app


@pytest.fixture
def client():
  application = create_app()
  application.config["TESTING"] = True
  with application.test_client() as c:
    yield c
