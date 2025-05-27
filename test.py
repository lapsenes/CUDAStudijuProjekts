import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Hyperparameters
INPUT_DIM    = 784    # 28×28 pixels flattened
HIDDEN_DIM   = 128
OUTPUT_DIM   = 10
EPOCHS       = 1000
LR           = 0.01
TRAIN_RATIO  = 0.8
CLIP_VALUE   = 5.0
WEIGHT_DECAY = 1e-4   # L2 regularization
SEED         = 42

# 1) Load the full MNIST tensor data once
dataset = datasets.MNIST('data', train=True, download=True,
                         transform=transforms.ToTensor())
full_X = dataset.data.view(-1, INPUT_DIM).float().div(255.0)  # [N,784], floats in [0,1]
full_y = dataset.targets                                  # [N]

# 2) Shuffle & split indices
torch.manual_seed(SEED)
perm = torch.randperm(len(full_X))
split = int(len(full_X) * TRAIN_RATIO)
train_idx, test_idx = perm[:split], perm[split:]

# 3) Create our big train/test tensors and move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

X_train = full_X[train_idx].to(device)
y_train = full_y[train_idx].to(device)
X_test  = full_X[test_idx].to(device)
y_test  = full_y[test_idx].to(device)

# 4) Define model
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.act = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)

model     = TwoLayerNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# 5) Training loop (full-batch)
torch.cuda.synchronize()
t0 = time.time()

for epoch in range(1, EPOCHS+1):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)                 # [train_size,10]
    loss   = criterion(logits, y_train)
    loss.backward()
    nn.utils.clip_grad_value_(model.parameters(), CLIP_VALUE)
    optimizer.step()

    # log every 10 epochs + last
    if epoch % 10 == 0 or epoch == EPOCHS:
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        avg = elapsed / epoch
        with torch.no_grad():
            preds = model(X_train).argmax(dim=1)
            acc   = (preds == y_train).float().mean().item()
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
              f"Acc: {acc:.4f} | Elapsed: {elapsed:.1f}s | "
              f"Avg/epoch: {avg:.3f}s")

torch.cuda.synchronize()
total_time = time.time() - t0
print(f"\nTraining complete in {total_time:.1f}s "
      f"({total_time/EPOCHS:.3f}s per epoch)\n")

# 6) Final test
model.eval()
with torch.no_grad():
    logits = model(X_test)
    test_acc = (logits.argmax(dim=1) == y_test).float().mean().item()
print(f"Test Accuracy: {test_acc:.4f}")
