import torch

device_name = ""  # Initialize Device name

if torch.cuda.is_available():
    device = torch.device("cuda")
    # Get the name of the current GPU
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print("Using CUDA GPU.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon MPS"  # MPS doesn't have a specific name function
    print("Using Apple Silicon MPS.")
else:
    device = torch.device("cpu")
    device_name = "CPU"  # Simple name for CPU
    print("Using CPU.")

# Print the full name of the selected device
print(f"Full Device Name: {device_name}")
print(f"Selected device object: {device}")
