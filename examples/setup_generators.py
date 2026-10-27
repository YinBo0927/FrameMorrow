"""Fetch pinned upstream source and official generator weights."""
import argparse
import json
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
BACKENDS = Path(__file__).with_name("backends.json")


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def checkout(root, config):
    if not root.exists():
        root.mkdir(parents=True)
        git(root, "init")
        git(root, "remote", "add", "origin", config["git"])
    if not (root / ".git").is_dir():
        raise ValueError(f"existing backend is not a Git checkout: {root}")
    if git(root, "remote", "get-url", "origin") != config["git"]:
        raise ValueError(f"backend remote differs from backends.json: {root}")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode == 0:
        if head.stdout.strip() != config["commit"] or git(root, "diff", "HEAD", "--name-only"):
            raise ValueError(f"preserve the existing backend; use a clean --sources-dir: {root}")
    else:
        if config.get("paths") or config["subdirectory"]:
            git(root, "sparse-checkout", "init", "--cone")
            git(root, "sparse-checkout", "set", *(config.get("paths") or [config["subdirectory"]]))
        git(root, "fetch", "--depth", "1", "--filter=blob:none", "origin", config["commit"])
        git(root, "checkout", "--detach", config["commit"])
    return root / config["subdirectory"]


def link(source, destination):
    source = Path(source).resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(f"preserving existing weights: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("all", "self-forcing", "longlive"), default="all")
    parser.add_argument("--sources-dir", type=Path, default=PROJECT / "third_party")
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--wan-dir", type=Path, help="reuse an existing Wan2.1-T2V-1.3B directory")
    parser.add_argument("--self-forcing-checkpoint", type=Path)
    parser.add_argument("--longlive-base-checkpoint", type=Path)
    parser.add_argument("--longlive-lora-checkpoint", type=Path)
    args = parser.parse_args()
    config = json.loads(BACKENDS.read_text())
    names = ("self-forcing", "longlive") if args.backend == "all" else (args.backend,)
    roots = {name: checkout(args.sources_dir.resolve() / name, config[name]) for name in names}
    if args.code_only:
        print(json.dumps({name: str(path) for name, path in roots.items()}, indent=2))
        return
    from huggingface_hub import hf_hub_download, snapshot_download
    wan = args.wan_dir
    if wan is None:
        wan = next((root / "wan_models/Wan2.1-T2V-1.3B" for root in roots.values()
                    if (root / "wan_models/Wan2.1-T2V-1.3B").is_dir()), None)
    if wan is None:
        wan = Path(snapshot_download(config["wan"]["repo"], revision=config["wan"]["revision"],
            allow_patterns=["config.json", "diffusion_pytorch_model*.safetensors", "Wan2.1_VAE.pth",
                            "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/*"]))
    for filename in ("config.json", "diffusion_pytorch_model.safetensors", "Wan2.1_VAE.pth",
                     "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/spiece.model"):
        if not (wan / filename).is_file():
            raise FileNotFoundError(f"missing native Wan model file: {wan / filename}")
    for name, root in roots.items():
        entry = config[name]
        link(wan, root / "wan_models/Wan2.1-T2V-1.3B")
        override = args.self_forcing_checkpoint if name == "self-forcing" else args.longlive_base_checkpoint
        destination = root / entry.get("checkpoint_target", entry["checkpoint"])
        checkpoint = override or (destination if destination.is_file() else Path(hf_hub_download(
            entry["weights_repo"], entry["checkpoint"], revision=entry["weights_revision"])))
        link(checkpoint, destination)
        if name == "longlive":
            destination = root / entry["lora_target"]
            lora = args.longlive_lora_checkpoint or (destination if destination.is_file()
                else Path(hf_hub_download(entry["weights_repo"], entry["lora"], revision=entry["weights_revision"])))
            link(lora, destination)
        print(f"Ready: {name} ({root})", flush=True)


if __name__ == "__main__":
    main()
