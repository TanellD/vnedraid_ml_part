from huggingface_hub import snapshot_download

checkpoints_path = snapshot_download(
    repo_id="levihsu/OOTDiffusion",
    allow_patterns=["checkpoints/**"],
    local_dir="./OOTDiffusion/checkpoints",
    local_dir_use_symlinks=False
)
