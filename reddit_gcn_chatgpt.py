import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

# Load dataset
dataset = Reddit(root='data/Reddit')
data = dataset[0]

# Define model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = GCN(dataset.num_features, 128, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Neighbor loader
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=1024,
    input_nodes=data.train_mask,
)

val_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=1024,
    input_nodes=data.val_mask,
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=1024,
    input_nodes=data.test_mask,
)

@torch.no_grad()
def evaluate(loader, desc="Evaluating"):
    model.eval()
    total_correct = 0
    total_examples = 0
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = out[:batch.batch_size].argmax(dim=-1)
        
        total_correct += int((pred == batch.y[:batch.batch_size]).sum())
        total_examples += batch.batch_size
    return total_correct / total_examples

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)

        loss = F.cross_entropy(
            out[:batch.batch_size],
            batch.y[:batch.batch_size]
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    val_acc = evaluate(val_loader, desc=f"Val Epoch {epoch}")
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Val Accuracy: {val_acc:.4f}\n")

print("\nTraining complete! Running final evaluation...")
test_acc = evaluate(test_loader, desc="Testing Model")
print(f"Final Test Accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), "gcn_reddit_model.pt")
print("Model saved to gcn_reddit_model.pt!")