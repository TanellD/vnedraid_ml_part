# Download CLIP model to checkpoints directory
from huggingface_hub import snapshot_download
import os

# Create the directory
os.makedirs('OOTDiffusion/checkpoints/clip-vit-large-patch14', exist_ok=True)

# Download CLIP model
print(' Downloading CLIP model...')
snapshot_download(
    repo_id='openai/clip-vit-large-patch14',
    local_dir='OOTDiffusion/checkpoints/clip-vit-large-patch14',
    local_dir_use_symlinks=False
)
