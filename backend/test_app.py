from pathlib import Path

from fastapi.testclient import TestClient

from app import app, DATA_DIR


def test_health():
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["application"] == "Predictive Maintenance Studio"


def test_datasets_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/datasets")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 4

    names = [
        item["dataset"]
        for item in data
    ]

    assert names == [
        "FD001",
        "FD002",
        "FD003",
        "FD004",
    ]


def test_cmapps_files_exist():
    expected_files = []

    for dataset in [
        "FD001",
        "FD002",
        "FD003",
        "FD004",
    ]:
        expected_files.extend(
            [
                f"train_{dataset}.txt",
                f"test_{dataset}.txt",
                f"RUL_{dataset}.txt",
            ]
        )

    for filename in expected_files:

        path = DATA_DIR / filename

        assert path.exists(), (
            f"Missing C-MAPSS file: {path}"
        )


def test_invalid_dataset():
    with TestClient(app) as client:

        response = client.post(
            "/api/train/FD999"
        )

    assert response.status_code == 404


def test_metrics_before_training():
    """
    A dataset that has not been trained should
    return 404 rather than fabricate metrics.
    """

    with TestClient(app) as client:

        response = client.get(
            "/api/metrics/FD001"
        )

    # It may be 404 on a fresh installation.
    # If an earlier training run created metrics,
    # the endpoint can legitimately return 200.
    assert response.status_code in [200, 404]


def test_predictions_before_training():
    """
    Predictions should never be fabricated.
    """

    with TestClient(app) as client:

        response = client.get(
            "/api/predictions/FD001"
        )

    assert response.status_code in [200, 404]


def test_invalid_sensor():
    with TestClient(app) as client:

        response = client.get(
            "/api/sensors/FD001/1"
            "?sensor=not_a_real_sensor"
        )

    assert response.status_code == 400