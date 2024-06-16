import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(123)

device = "mps" if torch.backends.mps.is_available() else "cpu"


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
    
#DoRA
class LinearWithDoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()
        self.linear = linear
        self.magnitude = nn.Parameter(self.linear.weight.norm(p=2, dim=0, keepdim=True))
        self.lora_layer = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha
        )

    def forward(self, x):
        lora_weights = self.lora_layer.alpha * (self.lora_layer.A @ self.lora_layer.B) #[IN_F, OUT_F]
        linear_weights = self.linear.weight #[OUT_F, IN_F]
        numerator = linear_weights + lora_weights.T #[OUT_F, IN_F
        denominator = numerator.norm(p=2, dim=0, keepdim=True)
        merged_weights = self.magnitude * numerator/denominator
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
    linear_layer = nn.Linear(50, 10)
    x = torch.randn((1, 50))

    print(f"Original output: {linear_layer(x)}")

    lora_layer_1 = LinearWithDoRAMerged(linear_layer, rank=2, alpha=4)
    print(f"LoRA output: {lora_layer_1(x)}")

    lora_layer_2 = LinearWithDoRAMerged(linear_layer, rank=2, alpha=4)
    print(f"LoRA merged output : {lora_layer_2(x)}")

def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)

# Generating synthetic data
def generate_data(num_samples=100, input_dim=10):
    X = torch.randn(num_samples, input_dim).to(device)
    y = torch.sum(X, dim=1, keepdim=True).to(device)  # Simple relationship for demonstration
    return X, y

def train(model, criterion, optimizer, data_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")



if __name__ == "__main__":
    
    model = MultiLayerPerceptron(
            num_features=128,
            num_hidden_1=256,
            num_hidden_2=512, 
            num_output=1
            )
    
    #adding lora layers to linear layers
    model.layers[0] = LinearWithDoRAMerged(model.layers[0], rank=4, alpha=8)
    model.layers[2] = LinearWithDoRAMerged(model.layers[2], rank=4, alpha=8)
    model.layers[4] = LinearWithDoRAMerged(model.layers[4], rank=4, alpha=8)

    freeze_linear_layers(model)
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    model.to(device)

    X, y = generate_data(num_samples=1000, input_dim=128)
    #X.to(device)
    #y.to(device)

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    train(model, criterion, optimizer, data_loader, epochs=500)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(data_loader))
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        print(f"Final Evaluation Loss: {loss.item()}")