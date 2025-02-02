import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class A(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(A, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, *, x):
        return self.linear(x)

class B(nn.Module):
    def __init__(self, dim=-1):
        super(B, self).__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, *, y):
        return self.softmax(y)

class C(nn.Module):
    def __init__(self, input_dim, output_dim, dim=-1):
        super(C, self).__init__()
        self.a = A(input_dim, output_dim)
        self.b = B(dim)

    def forward(self, *, z):
        return self.a(x=z)

# Training C with synthetic data
if __name__ == "__main__":
    input_dim, output_dim = 10, 5
    model_c = C(input_dim, output_dim)
    model_c = nn.DataParallel(model_c)  # Wrap in DataParallel
    model_c = model_c.cuda()
    
    # Create synthetic dataset
    X_train = torch.randn(100, input_dim).cuda()
    y_train = torch.randn(100, output_dim).cuda()
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_c.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model_c(z=batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    print("Training complete.")
