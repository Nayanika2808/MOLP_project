import pytest
from src.main import model

def test_model_training():
    # Verify that the model coefficients are not None after training
    assert model.coef_ is not None, "Model coefficients should not be None."
