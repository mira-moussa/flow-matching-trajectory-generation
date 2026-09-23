
import torch
import torch.nn as nn
import os as os

# Force the script to use the GPU
device = torch.device("cuda")

class HistoryEncoder(nn.Module):
    """
    Encodes past agent history tensor: 
    Input Shape:  (batch_size, 10, 5) -> [10 past frames, 5 features each]
    Output Shape: (batch_size, context_dim) -> [Flattened context vector]
    """
    def __init__(self, in_features=5, hidden_dim=64, context_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            # 1. Reshape the 2D matrix (10 frames x 5 features) into a single 1D vector of 50 features
            nn.Flatten(), 
            
            # 2. First hidden layer processing the 50 elements
            nn.Linear(10 * in_features, hidden_dim),
            nn.SiLU(),
            
            # 3. Output layer projecting down to the chosen context size (128)
            nn.Linear(hidden_dim, context_dim),
            nn.SiLU()
        )

    def forward(self, history):
        # history shape: (batch_size, 10, 5)
        return self.encoder(history)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x is shape (batch_size,) containing floats t in [0, 1]
        half_dim = self.dim // 2
        
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(x.device)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class VectorFieldNetwork(nn.Module):
    def __init__(self, traj_dim=60, time_dim=32, context_dim=128, hidden_dim=256):
        super().__init__()

        self.history_encoder= HistoryEncoder(context_dim=context_dim)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        self.net = nn.Sequential(
            nn.Linear(traj_dim + time_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, traj_dim) # Outputs velocity vector (vx, vy)
        )

    def forward(self, x_t, t, context):
        t=t.cuda()
        # x_t shape: (batch_size, 2)
        # t shape: (batch_size,)
        c_emb= self.history_encoder(context)
        t_emb = self.time_mlp(t.to(x_t.device))
        
        
        in_feat = torch.cat([x_t, t_emb, c_emb], dim=-1)
        return self.net(in_feat)

def compute_cfm_loss(model, x0, x1, history):
    batch_size = x0.shape[0]
    
    # 1. Sample uniform time t in [0, 1] for each item in batch

    t = torch.rand(batch_size).to(x0.device) 
    
    # Expand t for broadcasting with coordinates (batch_size, 1)
    t_expanded = t.unsqueeze(-1)
    t=t.cuda()
    # 2. Linear interpolation: x_t = (1 - t)*x0 + t*x1
    x_t = (1.0 - t_expanded) * x0 + t_expanded * x1

    x_t=x_t.cuda()
    # 3. Target velocity vector along straight line
    target_velocity = x1 - x0
    target_velocity=target_velocity.cuda()
    # 4. Neural network prediction
    pred_velocity = model(x_t, t,history)
    
    # 5. MSE Loss
    loss = torch.mean((pred_velocity - target_velocity) ** 2)
    return loss



def generate_trajectories_euler(model, num_samples=200, num_steps=50):
    model.eval()
    
    # 1. Start with fresh Gaussian noise at t=0
    x_t = torch.randn(num_samples, 2, device=device)
    dt = 1.0 / num_steps
    
    # Store trajectory history for visualization
    trajectory_history = [x_t.cuda().clone()]
    
    # 2. Integrate ODE from t=0 to t=1
    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((num_samples,), t_val, device=device)
        
        # Predict velocity field
        velocity = model(x_t, t_tensor)
        
        # Euler step forward
        x_t = x_t + velocity * dt
        trajectory_history.append(x_t.cpu().clone())
        
    # Shape: (num_steps + 1, num_samples, 2)
    return torch.stack(trajectory_history)