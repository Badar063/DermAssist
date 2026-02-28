import torch
from src.models.cnn import create_model

def test_model_output_shape():
    model = create_model('simple', num_classes=3)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 3)
