import torch.nn as nn
from typing import Optional, Union
import torch
class LoRAConv1DLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,

        original_layer: nn.Module,

        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()


        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels

        self.lora_a = nn.Conv1d(self.in_channels, rank, kernel_size=1, device=device, dtype=dtype)
        self.lora_b = nn.Conv1d(rank, self.out_channels,  kernel_size=1, device=device, dtype=dtype)


        self.original_layer = original_layer
        self.network_alpha = network_alpha
        self.rank = rank


        nn.init.normal_(self.lora_a.weight, std=1 / rank)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.lora_a.weight.dtype

        down_hidden_states = self.lora_a(hidden_states.to(dtype))
        up_hidden_states = self.lora_b(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        output = up_hidden_states.to(orig_dtype) + self.original_layer(hidden_states)

        return output

class LoRALinearLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,

        original_layer: nn.Module,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()


        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        self.lora_a = nn.Linear(self.in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(rank, self.out_features, bias=False, device=device, dtype=dtype)


        self.original_layer = original_layer
        self.network_alpha = network_alpha
        self.rank = rank


        nn.init.normal_(self.lora_a.weight, std=1 / rank)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.lora_a.weight.dtype

        down_hidden_states = self.lora_a(hidden_states.to(dtype))
        up_hidden_states = self.lora_b(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        output = up_hidden_states.to(orig_dtype) + self.original_layer(hidden_states)



        return output



def freeze_original_parameters(model, adapter=LoRALinearLayer):
    for name, param in model.named_parameters():

            param.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, adapter):
                module.lora_a.weight.requires_grad = True

                module.lora_b.weight.requires_grad = True


def reverse_freeze_original_parameters(model):
    for name, param in model.named_parameters():

            param.requires_grad = True

