import kagglehub

# Download latest version
path = kagglehub.dataset_download("andychamberlain/the-lick")

print("Path to dataset files:", path)
