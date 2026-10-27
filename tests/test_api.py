from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["model"] == "Random Forest"


def test_valid_prediction():
    payload = {
        "machine_type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["risk_probability"] <= 1
    assert 0 <= result["risk_percentage"] <= 100
    assert isinstance(result["machine_failure"], bool)
    assert result["model"] == "Random Forest"


def test_invalid_machine_type():
    payload = {
        "machine_type": "X",
        "air_temperature": 300,
        "process_temperature": 310,
        "rotational_speed": 1500,
        "torque": 40,
        "tool_wear": 100,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_sensor_range():
    payload = {
        "machine_type": "M",
        "air_temperature": 100,
        "process_temperature": 310,
        "rotational_speed": 1500,
        "torque": 40,
        "tool_wear": 100,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
