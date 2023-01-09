import torch
from src.models.model import Model
import pytest

def test_model():
    # create a model instance
    model = Model()
    # check a given input should have the shape [784] and output should have the shape [10]
    assert model(torch.randn(1, 784)).shape == torch.Size([1, 10]), "Model output has the wrong shape"
    
# tests/test_model.py
# def test_error_on_wrong_shape():
#     model = Model()
#     with pytest.raises(ValueError, match='Expected input to a 1D tensor'):
#       model(torch.randn(1))
