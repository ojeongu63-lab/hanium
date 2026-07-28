def test_mlflow_importable():
    import mlflow

    assert mlflow.__version__


def test_mlflow_pytorch_importable():
    import mlflow.pytorch  # noqa: F401


def test_fastapi_importable():
    import fastapi

    assert fastapi.__version__


def test_uvicorn_importable():
    import uvicorn

    assert uvicorn.__version__
