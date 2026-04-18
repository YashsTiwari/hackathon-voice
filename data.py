from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="XuepingZhang/ESDD2-CompSpoof-V2",
    repo_type="dataset",
    local_dir="./CompSpoofV2",
    max_workers=8,
)