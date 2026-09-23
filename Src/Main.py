
from os import path
import torch
import DataGeneration
from torch.utils.data import DataLoader
from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
import numpy as np
import visualization
import Model
    



# Initialize nuScenes devkit 
# (Make sure you point to your local installation directory)
nusc = NuScenes(version='v1.0-mini', dataroot='D:/my_project/data/sets/nuscenes', verbose=False)

# Initialize dataset and loader
train_dataset = DataGeneration.NuScenesCFMDataset(nusc, split='mini_train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

# Initialize your CFM model
model = Model.VectorFieldNetwork(traj_dim=60, time_dim=32, context_dim=128, hidden_dim=256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)



# --- Training Loop ---
model.train()
for epoch in range(120):
    for batch_idx, (history, x1) in enumerate(train_loader):
        history = history.to(device) # (batch_size, 10, 5)
        x1 = x1.to(device)           # (batch_size, 60)
        
        # Draw standard Gaussian noise configurations x0 matching your trajectory dimensions
        x0 = torch.randn_like(x1).to(device) # (batch_size, 60)
        
        optimizer.zero_grad()
        
        # Compute loss using your flow matching objective
        loss = Model.compute_cfm_loss(model, x0, x1, history)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | CFM Loss: {loss.item():.4f}")



# Run it using your trained elements!
visualization.visualize_cfm_trajectory(model, train_dataset, index=42)



# Run the test block!
visualization.generate_test_trajectory(model, train_dataset, index=42)