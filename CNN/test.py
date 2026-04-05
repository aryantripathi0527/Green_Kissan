import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from cnn_model import BiomassCNN
from torch_dataset import BiomassDataset, compute_dataset_stats

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# Load dataset stats
ch_mean, ch_std, lb_mean, lb_std = compute_dataset_stats(
    'data/real_metadata.csv', 'data'
)

# Load model
ckpt = torch.load('checkpoints/cnn_biomass.pt', map_location=DEVICE)
model = BiomassCNN(in_channels=5, dropout_rate=0.5).to(DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print('Model loaded successfully')

# Dataset
ds = BiomassDataset(
    'data/real_metadata.csv',
    'data',
    augment=False,
    channel_mean=ch_mean,
    channel_std=ch_std,
    label_mean=lb_mean,
    label_std=lb_std
)

n = len(ds)
n_test = n - int(0.8 * n)

loader = DataLoader(
    Subset(ds, list(range(int(0.8 * n), n))),
    batch_size=16,
    shuffle=False
)

print(f'Test samples: {n_test}')

# Evaluation
preds, actuals = [], []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)

        try:
            out, _ = model(imgs)
        except:
            out = model(imgs)

        p = out.squeeze().cpu().numpy() * lb_std + lb_mean
        a = labels.numpy() * lb_std + lb_mean

        preds.extend(p.flatten())
        actuals.extend(a.flatten())

preds = np.array(preds)
actuals = np.array(actuals)

# Metrics
mae = np.mean(np.abs(preds - actuals))
mse = np.mean((preds - actuals) ** 2)
rmse = np.sqrt(mse)

ss_res = np.sum((actuals - preds) ** 2)
ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
r2 = 1 - ss_res / ss_tot

mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-6))) * 100
acc = 100 - mape

# Output
print()
print('=' * 45)
print(f'  MSE      : {mse:.4f} (t/ha)^2')
print(f'  RMSE     : {rmse:.4f} t/ha')
print(f'  MAE      : {mae:.4f} t/ha')
print(f'  R2 Score : {r2:.4f}')
print(f'  MAPE     : {mape:.2f}%')
print(f'  Accuracy : {acc:.2f}%')
print('=' * 45)
print()

print('Per-sample preview (first 10):')
print(f'  {"Actual":>10}  {"Predicted":>10}  {"Error":>10}')
print(f'  {"-"*10}  {"-"*10}  {"-"*10}')

for i in range(min(10, len(actuals))):
    err = abs(actuals[i] - preds[i])
    print(f'  {actuals[i]:>10.4f}  {preds[i]:>10.4f}  {err:>10.4f}')