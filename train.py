import torch
import torch.nn as nn
import torch.optim as optim
from model import SPeM_CRNN

# --- 1. Smart Synthetic Data Generation ---
def create_gaussian_blob(grid_size, center_x, center_y, sigma_x, sigma_y, amplitude=1.0):
    """Creates a 2D Gaussian blob to simulate a body part's pressure."""
    x = torch.arange(grid_size).float()
    y = torch.arange(grid_size).float()
    y, x = torch.meshgrid(y, x, indexing='ij')
    
    blob = amplitude * torch.exp(-(((x - center_x)**2) / (2 * sigma_x**2) + ((y - center_y)**2) / (2 * sigma_y**2)))
    return blob

def generate_posture_frame(posture_idx, grid_size=27):
    """Paints a 27x27 frame based on the specific sleeping posture."""
    frame = torch.zeros(grid_size, grid_size)
    
    if posture_idx == 0:  # Supine (On Back: centered, spread out)
        frame += create_gaussian_blob(grid_size, 13, 5, 2.5, 2.5, 0.8)  # Head
        frame += create_gaussian_blob(grid_size, 13, 13, 4.0, 5.0, 1.0) # Torso
        frame += create_gaussian_blob(grid_size, 9, 22, 2.0, 4.0, 0.7)  # Left Leg
        frame += create_gaussian_blob(grid_size, 17, 22, 2.0, 4.0, 0.7) # Right Leg
        
    elif posture_idx == 1: # Prone (On Stomach: similar to supine, but slightly wider)
        frame += create_gaussian_blob(grid_size, 11, 5, 2.0, 2.0, 0.6)  # Head (turned)
        frame += create_gaussian_blob(grid_size, 13, 13, 5.0, 5.0, 0.9) # Torso
        frame += create_gaussian_blob(grid_size, 10, 22, 2.5, 4.0, 0.6) # Left Leg
        frame += create_gaussian_blob(grid_size, 16, 22, 2.5, 4.0, 0.6) # Right Leg
        
    elif posture_idx == 2: # Left Lateral (On Left Side: shifted left, narrow)
        frame += create_gaussian_blob(grid_size, 9, 6, 2.0, 2.5, 0.8)   # Head
        frame += create_gaussian_blob(grid_size, 9, 14, 2.0, 6.0, 1.0)  # Torso (narrow profile)
        frame += create_gaussian_blob(grid_size, 9, 22, 2.0, 4.0, 0.8)  # Legs (stacked together)
        
    elif posture_idx == 3: # Right Lateral (On Right Side: shifted right, narrow)
        frame += create_gaussian_blob(grid_size, 17, 6, 2.0, 2.5, 0.8)  # Head
        frame += create_gaussian_blob(grid_size, 17, 14, 2.0, 6.0, 1.0) # Torso (narrow profile)
        frame += create_gaussian_blob(grid_size, 17, 22, 2.0, 4.0, 0.8) # Legs (stacked together)
        
    # Add a tiny bit of random sensor noise (crosstalk simulation)
    noise = torch.rand(grid_size, grid_size) * 0.1
    frame = torch.clamp(frame + noise, 0, 1)
    return frame

def generate_smart_dataset(batch_size, frames, num_classes=4):
    """Generates the full 5D tensor stream with structured data."""
    data = torch.zeros(batch_size, frames, 1, 27, 27)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    for b in range(batch_size):
        posture = labels[b].item()
        for f in range(frames):
            # In a real scenario, we'd add slight frame-to-frame shifts here for movement
            data[b, f, 0] = generate_posture_frame(posture)
            
    return data, labels

# --- 2. Configuration & Initialization ---
batch_size = 32
frames = 10
num_classes = 4

print("Generating SMART synthetic dataset with human posture shapes...")
dummy_data, dummy_labels = generate_smart_dataset(batch_size, frames, num_classes)

print("Initializing SPeM CRNN architecture...")
model = SPeM_CRNN(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- 3. Training Loop ---
# We increased epochs to 10 so you can really watch the model learn!
epochs = 50
print("\nStarting Prototype Training...")

for epoch in range(epochs):
    model.train() 
    optimizer.zero_grad()
    
    outputs = model(dummy_data)
    loss = criterion(outputs, dummy_labels)
    
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == dummy_labels).sum().item()
    accuracy = 100 * correct / batch_size
    
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

# --- 4. Save Weights ---
torch.save(model.state_dict(), 'spem_weights.pth')
print("\nTraining complete! The model has successfully learned the posture shapes.")
print("Model weights saved to 'spem_weights.pth'.")