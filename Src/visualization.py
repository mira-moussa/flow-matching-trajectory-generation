import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def visualize_cfm_trajectory(model, dataset, index=0):
    """
    Plots the vehicle's actual history path, true future path,
    and visualizes how CFM generates velocity fields to bridge them.
    Supports both CPU and GPU execution automatically.
    """
    # 1. Fetch a real sample from your fixed data loader
    history, x1_true = dataset[index]
    
    # Add batch dimension for your PyTorch model [1, 10, 5] and [1, 60]
    history_batch = history.unsqueeze(0) 
    x1_true_batch = x1_true.unsqueeze(0)
    
    # 2. Reshape raw 1D flattened coordinates back to 2D paths (X, Y)
    # The past has 10 frames of (x, y) at indices 0 and 1
    past_path = history[:, :2].numpy() 
    # The true future path has 30 coordinate pairs (flattened to 60 elements)
    true_future_path = x1_true_batch.view(30, 2).detach().numpy()
    
    # 3. Create sample paths at different Flow Matching timesteps (t)
    # At t=0, it is pure Gaussian noise (where path generation starts)
    x0_noise = torch.randn_like(x1_true_batch)
    
    # Detect which device the model is living on (e.g., 'cuda:0' or 'cpu')
    device = next(model.parameters()).device
    model.eval()
    
    timesteps = [0.1, 0.5, 0.9]
    morphed_paths = {}
    
    with torch.no_grad():
        for t_val in timesteps:
            # Create tensors and explicitly move them to the GPU/Model device
            t_tensor = torch.tensor([t_val], dtype=torch.float32).to(device).view(1)
            
            # Flow matching interpolation formula: xt = (1 - t)*x0 + t*x1
            xt = (1 - t_val) * x0_noise + t_val * x1_true_batch
            xt = xt.to(device)
            
            # Move the context data to the same device
            history_inputs = history_batch.to(device)
            
            # Predict the velocity vectors using your trained network model!
            predicted_velocity = model(xt, t_tensor, history_inputs)
            
            # Move data BACK to cpu memory so numpy/matplotlib can process it
            morphed_paths[t_val] = {
                'positions': xt.view(30, 2).cpu().numpy(),
                'velocities': predicted_velocity.view(30, 2).cpu().numpy()
            }

    # --- PLOTTING CODE ---
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Vehicle Past History (Blue Dots moving into the present)
    plt.plot(past_path[:, 0], past_path[:, 1], 'o-', color='royalblue', label='Vehicle Past History (2s)', markersize=6)
    # Mark the current ego vehicle location (0,0 point in local coordinates)
    plt.plot(0, 0, 'X', color='red', markersize=12, label='Current Vehicle Position (t=0)')
    
    # Plot 2: Ground Truth Future Target Path (Solid Green Line)
    plt.plot(true_future_path[:, 0], true_future_path[:, 1], 'g-', linewidth=3, label='True Future Path Ground Truth (6s)')
    
    # Plot 3: Draw the velocity field lines showing how noise converts to reality
    # We use quiver to draw actual mathematical directional arrows!
    colors = ['purple', 'orange', 'crimson']
    for idx, t_val in enumerate(timesteps):
        pos = morphed_paths[t_val]['positions']
        vel = morphed_paths[t_val]['velocities']
        
        # Draw arrows representing the vector fields matching targets
        plt.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], 
                   color=colors[idx], alpha=0.6, scale=50,
                   label=f'Learned CFM Vector Field (t={t_val})')
        
    plt.title('NuScenes Vehicle Trajectory & CFM Flow Fields Visualization', fontsize=14, fontweight='bold')
    plt.xlabel('X coordinate (Meters relative to vehicle)', fontsize=12)
    plt.ylabel('Y coordinate (Meters relative to vehicle)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal') # Keep scales identical so geometric curves aren't stretched
    plt.legend(loc='best')
    
    # Save chart locally
    plt.savefig('vehicle_trajectory_flow_map.png', dpi=300)
    print("Saved beautiful custom field map to 'vehicle_trajectory_flow_map.png'!")
    plt.show()

def generate_test_trajectory(model, dataset, index=0, steps=60):
    """
    TESTING PATH: Generates a completely new vehicle trajectory 
    using ONLY past history and random noise (Euler integration).
    """
    # 1. Grab history context
    history, x1_true = dataset[index]
    history_batch = history.unsqueeze(0) # [1, 10, 5]
    
    device = next(model.parameters()).device
    model.eval()
    
    # 2. START OF TESTING: Sample absolute pure random noise at t=0
    # The model has to turn this chaos into a real driving path

    xt = torch.randn(1, 60, dtype=torch.float32).to(device) *0.2
    # 3. Euler integration loop (Moving from t=0 to t=1)
    dt = 1.0 / steps
    
    with torch.no_grad():
        for step in range(steps):
            # Calculate current normalized time scalar
            t_val = step * dt
            t_tensor = torch.tensor([t_val], dtype=torch.float32).to(device).view(1)
            
            # Predict velocity fields at this exact micro-moment
            predicted_velocity = model(xt, t_tensor, history_batch.to(device))
            
            # Move the path points forward a tiny bit along the velocity arrow
            # Formula: x_{t+dt} = x_t + v(x_t, t) * dt
            xt = xt + predicted_velocity * dt
            
    # 4. Process outputs back to readable coordinate pairs
    generated_path = xt.view(30, 2).cpu().numpy()
    x_coords = generated_path[:, 0]
    y_coords = generated_path[:, 1]
    x_smooth = gaussian_filter1d(x_coords, sigma=0.5)
    y_smooth = gaussian_filter1d(y_coords, sigma=0.5)
    generated_path_smooth = np.stack([x_smooth, y_smooth], axis=1)
    true_future_path = x1_true.view(30, 2).numpy()
    past_path = history[:, :2].numpy()
    
    # --- TEST COMPARISON PLOT ---
    plt.figure(figsize=(10, 6))
    plt.plot(past_path[:, 0], past_path[:, 1], 'o-', color='royalblue', label='Input Past History (2s)')
    plt.plot(0, 0, 'X', color='red', markersize=10, label='Current Car Position')
    
    plt.plot(true_future_path[:, 0], true_future_path[:, 1], 'g-', linewidth=2, label='True Future (What actually happened)')
    plt.plot(generated_path_smooth[:, 0], generated_path_smooth[:, 1], 'r--', linewidth=3, label='CFM Generated Test Path (Model Prediction)')
    
    plt.title('TEST MODE: Actual Model Trajectory Generation')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.axis('equal')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig('cfm_inference_test.png', dpi=300)
    print("Saved true testing evaluation plot to 'cfm_inference_test.png'!")
    plt.show()
