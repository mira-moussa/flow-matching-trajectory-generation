import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
import os as os
import json
# Force the script to use the GPU
device = torch.device("cuda")


def generate_obstacle_trajectories(num_samples=2000):
    """
    Generates synthetic 2D target points (x1) that bend left or right 
    around a circular obstacle centered at (0, 0).
    """
    # Half the paths go left (-1), half go right (+1)
    sides = torch.randint(0, 2, (num_samples,)).float() * 2.0 - 1.0
   
    # Radius around obstacle with a bit of noise
    radius = 1.5 + 0.1 * torch.randn(num_samples)
    
    # Angle sweep (from bottom -pi/2 to top pi/2)
    angles = (torch.rand(num_samples) - 0.5) * torch.pi
    
    # Compute x1 target points
    x_coords = sides * radius * torch.cos(angles)
    y_coords = 2.5 * torch.sin(angles) # scale vertically
    
    
    # Base distribution (x0) is standard Gaussian noise
    x0_x= torch.randn_like(x_coords)
    X0_y= torch.randn_like(y_coords)

    x0 = torch.stack([x0_x, X0_y], dim=1)
    x1 = torch.stack([x_coords, y_coords], dim=1)
    
    return x0, x1

# Quick visual check of our dataset
x0,x1 = generate_obstacle_trajectories(1000)

plt.figure(figsize=(6, 6))
plt.scatter(x0[:,0], x0[:,1], alpha=0.3, label="Base Noise (x0)", color="gray")
plt.scatter(x1[:,0], x1[:,1], alpha=0.5, label="Target Trajectories (x1)", color="crimson")
plt.scatter([0], [0], color="black", s=200, marker="X", label="Obstacle")
plt.title("Phase 1 Dataset: Base Noise vs. Multi-Modal Target")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

###############################################################################################
# Step 2: The Neural Vector Field & CFM Loss





class NuScenesCFMDataset(Dataset):
    def __init__(self, nusc: NuScenes, split: str = 'mini_train'):
        self.nusc = nusc
        self.helper = PredictHelper(nusc)
        # Fetch token pairs for the standard prediction challenge splits
       
        # Forces the path to use clean, uniform forward-slashes that python can read easily
       # --- BYPASS BROKEN DEVKIT PATH BUG ---
        
       
        
        # 1. Force the absolute path to your exact maps/prediction folder
        clean_dataroot = os.path.abspath(nusc.dataroot)
        json_path = os.path.join(clean_dataroot, 'maps', 'prediction', 'prediction_scenes.json')
        
        # 2. Check if the file is actually there before trying to read it
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"\n\n[ERROR] Could not find the file at: {json_path}\n"
                                    f"Please make sure you unzipped the full nuScenes dataset into your data folder!")

        # 3. Read the json data directly ourselves
        # --- UPDATE THIS LOGIC IN YOUR CUSTOM FILE LOADER ---
        with open(json_path, "r") as f:
            prediction_scenes = json.load(f)
            
        self.token_pairs = []
        for scene_id, tokens in prediction_scenes.items():
            for token_pair in tokens:
                # The pair string format is always: "instanceToken_sampleToken"
                instance_token, sample_token = token_pair.split('_')
                
                # --- SAFE GUARD FOR MINI DATASET ---
                # Check if this exact pair actually exists in your mini-dataset lookup map
                if (sample_token, instance_token) in self.helper.inst_sample_to_ann:
                    self.token_pairs.append(token_pair)
            
        print(f"Successfully filtered and loaded {len(self.token_pairs)} local mini-dataset trajectories!")
        # ----------------------------------------------------
            
       
        # --------------------------------------
        
        # Configuration matches your model specifications
        self.past_horizon = 10   # 2 seconds at 5Hz
        self.future_horizon = 30 # 6 seconds at 5Hz

    def __len__(self):
        return len(self.token_pairs)

    def __getitem__(self, idx):
        # 1. Extract tokens
        token_pair = self.token_pairs[idx]
        instance_token, sample_token = token_pair.split('_')
        
        # 2. Get past and future trajectories from the helper
        past_traj = self.helper.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=True)
        future_traj = self.helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
        
      
        
        # Pad or slice past trajectory to ensure it is exactly 10 frames, 5 features
        # (If past_traj is shape (N, 2), we pad with zeros for velocity/acceleration/heading features)
        history_tensor = torch.zeros(10, 5, dtype=torch.float32)
        for i in range(min(len(past_traj), 10)):
            history_tensor[i, 0] = float(past_traj[i][0]) # x
            history_tensor[i, 1] = float(past_traj[i][1]) # y
            # positions 2, 3, 4 remain 0 if not provided by this basic helper function
            
        # Pad or slice future trajectory to ensure it is exactly 30 frames (60 dimensions total)
        future_coords = []
        for i in range(30):
            if i < len(future_traj):
                future_coords.append(float(future_traj[i][0])) # x
                future_coords.append(float(future_traj[i][1])) # y
            else:
                # If the trajectory is too short, repeat the last known position to pad it
                future_coords.append(float(future_traj[-1][0] if len(future_traj) > 0 else 0.0))
                future_coords.append(float(future_traj[-1][1] if len(future_traj) > 0 else 0.0))
                
        x1 = torch.tensor(future_coords, dtype=torch.float32)
        
        return history_tensor, x1

