import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.models.embodiment.uninavid.uninavid_action_model import (
    UniNaVidForActionPrediction,
)


def test_pre_response_value_uses_final_hidden_before_response_start():
    model = UniNaVidForActionPrediction(
        tokenizer=None,
        model=nn.Identity(),
        image_processor=None,
    )
    model.value_head = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.value_head.weight.copy_(torch.tensor([[1.0, 10.0, 100.0]]))

    final_hidden = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ]
    )

    values = model._compute_value_from_pre_response_hidden(
        final_hidden=final_hidden,
        prompt_len=2,
    )

    assert values.shape == (1, 1)
    assert values.item() == 20.0


def test_uninavid_constructor_adds_value_head_from_config_hidden_size():
    class _Backbone(nn.Identity):
        config = type("Config", (), {"hidden_size": 3})()

    model = UniNaVidForActionPrediction(
        tokenizer=None,
        model=_Backbone(),
        image_processor=None,
        cfg=OmegaConf.create({"add_value_head": True}),
    )

    assert hasattr(model, "value_head")
    assert model.value_head(torch.ones(1, 3)).shape == (1, 1)


def test_pre_response_value_preserves_hidden_dtype_for_wrapped_value_head():
    class _WrappedValueHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nn.Linear(3, 1, bias=False).to(dtype=torch.bfloat16)
            self._reported_param = nn.Parameter(torch.ones(1, dtype=torch.float32))

        def parameters(self, recurse=True):
            yield self._reported_param

        def forward(self, x):
            return self.inner(x)

    model = UniNaVidForActionPrediction(
        tokenizer=None,
        model=nn.Identity(),
        image_processor=None,
    )
    model.value_head = _WrappedValueHead()
    final_hidden = torch.ones(1, 2, 3, dtype=torch.bfloat16)

    values = model._compute_value_from_pre_response_hidden(
        final_hidden=final_hidden,
        prompt_len=1,
    )

    assert values.shape == (1, 1)
    assert values.dtype == torch.float32
