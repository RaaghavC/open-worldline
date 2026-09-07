"""Fetch only WorldFM's core model and VAE, using the verified public revision."""
import argparse
from pathlib import Path

from fetch_assets import fetch


ARTIFACTS = [
    {"path": "worldfm_2-step.pth", "size": 2458577701,
     "sha256": "7cf8462ca3e9acaa0c7cf6f5b99917f2dd0b9d03dee9837237c28bf5700c7b17"},
    {"path": "vae/diffusion_pytorch_model.safetensors", "size": 334643238,
     "sha256": "1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c"},
    {"path": "vae/config.json", "size": 631,
     "sha256": "f18b16fa4381c90ab44a3d7abd2e90afd05a2c31b8ca69998bc93c6453aeb7b7"},
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fetch("inspatio/worldfm", "48d206b813a6ff3ddc1977b6cce8b6769c3bbad6", ARTIFACTS, args.output)
