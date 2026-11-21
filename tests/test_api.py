
import pytest
from fastapi.testclient import TestClient

from main import app


# Create a test client
@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# Test 1: Root endpoint
def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


# Test 2: Health endpoint
def test_health_endpoint(client):
    response = client.get("/health")
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "healthy"
    assert result["model_loaded"] is True


# Test 3: Model information endpoint
def test_model_info_endpoint(client):
    response = client.get("/model-info")
    result = response.json()

    assert response.status_code == 200
    assert result["model_type"] == "RandomForestClassifier"
    assert result["model_version"] == "1.0.0"
    assert result["test_accuracy"] == 0.9


# Test 4: Valid prediction
def test_valid_prediction(client):
    request_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=request_data)
    result = response.json()

    assert response.status_code == 200
    assert result["predicted_species"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]
    assert 0 <= result["confidence"] <= 1
    assert "probabilities" in result
    assert result["model_version"] == "1.0.0"


# Test 5: Negative feature value
def test_negative_feature_value(client):
    request_data = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=request_data)

    assert response.status_code == 422


# Test 6: Missing feature
def test_missing_feature(client):
    request_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post("/predict", json=request_data)

    assert response.status_code == 422


# Test 7: Incorrect data type
def test_incorrect_data_type(client):
    request_data = {
        "sepal_length": "invalid",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=request_data)

    assert response.status_code == 422


# Test 8: Unknown endpoint
def test_unknown_endpoint(client):
    response = client.get("/unknown")

    assert response.status_code == 404
