import torch
from torch_geometric.datasets import Reddit
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

# Load dataset
dataset = Reddit(root='data/Reddit')
data = dataset[0]

# Define same model
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = GCN(dataset.num_features, 128, dataset.num_classes).to(device)
model.load_state_dict(torch.load("gcn_reddit_model.pt", map_location=device, weights_only=True))
model.eval()

# Inference via NeighborLoader to avoid OOM
# Batching prevents allocating memory for the entire 114M+ edge graph all at once
loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # Same sampling size used during training
    batch_size=1024,
    shuffle=False            # Iterate through all nodes in original order
)

preds = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        # Take only the predictions for the root/target nodes in the batch
        pred = out[:batch.batch_size].argmax(dim=1)
        preds.append(pred.cpu())

pred = torch.cat(preds, dim=0)

print("Predictions for the first 10 nodes:", pred[:10])

correct = (pred == data.y)
accuracy = correct.sum().item() / len(correct)
print("Accuracy:", accuracy)