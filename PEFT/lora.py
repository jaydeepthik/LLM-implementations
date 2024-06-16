import torch
from torch import nn
import torch.nn.functional as F

#lora
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha) -> None:
        super().__init__()

        self.alpha = alpha
        self.std = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_features, rank)*self.std)
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()
        self.linear = linear
        self.lora_layer = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha
        )

    def forward(self, x):
        x = self.linear(x) + self.lora_layer(x)
        return x
    
class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()
        self.linear = linear
        self.lora_layer = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha
        )

    def forward(self, x):
        lora_weights = self.lora_layer.alpha * (self.lora_layer.A @ self.lora_layer.B) #[IN_F, OUT_F]
        linear_weights = self.linear.weight #[OUT_F, IN_F]

        merged_weights = linear_weights + lora_weights.T #[OUT_F, IN_F]
        return F.linear(x, merged_weights, self.linear.bias)

#MLP
class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_features, num_hidden_1, num_hidden_2, num_output) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2, num_output)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

def test_lora():
    torch.manual_seed(123)
    linear_layer = nn.Linear(50, 10)
    x = torch.randn((1, 50))

    print(f"Original output: {linear_layer(x)}")

    lora_layer_1 = LinearWithLoRA(linear_layer, rank=2, alpha=4)
    print(f"LoRA output: {lora_layer_1(x)}")

    lora_layer_2 = LinearWithLoRAMerged(linear_layer, rank=2, alpha=4)
    print(f"LoRA merged output : {lora_layer_2(x)}")

def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)

if __name__ == "__main__":
    
    model = MultiLayerPerceptron(
            num_features=128,
            num_hidden_1=256,
            num_hidden_2=512, 
            num_output=10
            )

    #adding lora layers to linear layers
    model.layers[0] = LinearWithLoRA(model.layers[0], rank=4, alpha=8)
    model.layers[2] = LinearWithLoRA(model.layers[2], rank=4, alpha=8)
    model.layers[4] = LinearWithLoRA(model.layers[4], rank=4, alpha=8)

    freeze_linear_layers(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")