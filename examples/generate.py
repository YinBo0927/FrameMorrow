"""Shared native-weight video generation for the two runnable examples."""
import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]


def main(backend):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    settings = json.loads((PROJECT / "configs/default.json").read_text())
    parser.add_argument("--k", type=int, default=settings["inference"]["k"])
    parser.add_argument("--backend-root", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs" / backend)
    parser.add_argument("--prompt", default="A small red toy car drives slowly on a wooden table, soft daylight, static camera.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--chunks", type=int, default=11)
    parser.add_argument("--warmup-chunks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--visual-model", default="facebook/dinov2-base")
    parser.add_argument("--condition-model", default="google/umt5-xxl")
    parser.add_argument("--compare-baseline", action="store_true")
    args = parser.parse_args()
    if (min(args.height, args.width) < 16 or args.height % 16 or args.width % 16
            or args.warmup_chunks < 6 or args.chunks <= args.warmup_chunks or args.threads < 1
            or not 1 <= args.k <= 4):
        parser.error("use dimensions divisible by 16, warmup-chunks >= 6, chunks > warmup-chunks, threads >= 1, k in 1..4")
    if not args.prompt.strip():
        parser.error("prompt cannot be empty")
    if args.device == "cpu":
        # Upstream LongLive automatically moves its text encoder to CUDA when visible.
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            os.environ[name] = ""
    import numpy as np
    from PIL import Image
    import torch
    from functools import partial
    sys.path.insert(0, str(PROJECT))
    from framemorrow import (FrameMorrow, SelectorConfig, SelfForcingHistoryAdapter,
                            LongLiveHistoryAdapter)
    from framemorrow.encoders import FrozenEncoders

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    config = json.loads(Path(__file__).with_name("backends.json").read_text())[backend]
    root = (args.backend_root or PROJECT / "third_party" / backend / config["subdirectory"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != config["commit"]:
        raise ValueError("backend revision differs from examples/backends.json; rerun setup in a clean directory")
    output, checkpoint_path = args.output.resolve(), args.selector_checkpoint.resolve(strict=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("visual_model", "condition_model"):
        value = Path(getattr(args, name))
        if value.exists():
            setattr(args, name, str(value.resolve()))
    os.chdir(root)
    sys.path.insert(0, str(root))
    started = time.monotonic()

    def log(message):
        print(f"[{time.monotonic() - started:.1f}s] {message}", flush=True)

    # Upstream T5 evaluates a CUDA default at import time, even for CPU inference.
    original_device = torch.cuda.current_device
    try:
        if device.type == "cpu":
            torch.cuda.current_device = lambda: 0
        from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
    finally:
        torch.cuda.current_device = original_device
    attention = importlib.import_module("wan.modules.attention")
    attention.FLASH_ATTN_2_AVAILABLE = attention.FLASH_ATTN_3_AVAILABLE = False
    modules = ["wan.modules.model", "wan.modules.causal_model"]
    if backend == "longlive":
        modules.append("wan.modules.causal_model_latentmem")
    for name in modules:
        module = importlib.import_module(name)
        for attribute in ("attention", "flash_attention"):
            if callable(getattr(module, attribute, None)):
                setattr(module, attribute, partial(attention.attention, dtype=dtype))

    log(f"Loading official {backend} generator, native text encoder and VAE")
    memory_kwargs = {"use_latentmem": True, "memory_size": args.k} if backend == "longlive" else {}
    generator = WanDiffusionWrapper(is_causal=True, local_attn_size=config["local_frames"],
                                    sink_size=config["sink_frames"], timestep_shift=5., **memory_kwargs)
    generator_path = root / config.get("checkpoint_target", config["checkpoint"])
    state = torch.load(generator_path, map_location="cpu", weights_only=False, mmap=True)
    keys = ("generator_ema", "generator") if backend == "self-forcing" else ("generator", "model")
    weights = next((state[key] for key in keys if key in state), None)
    if weights is None:
        raise ValueError("official checkpoint has no generator state")
    generator.load_state_dict({key.replace("_fsdp_wrapped_module.", ""): value for key, value in weights.items()}, strict=True)
    del weights, state
    if backend == "longlive":
        import peft
        from omegaconf import OmegaConf
        from utils.lora_utils import configure_lora_for_model
        native_config = OmegaConf.load(root / config["native_config"])
        generator.model = configure_lora_for_model(generator.model, "generator", native_config.adapter,
                                                   is_main_process=False)
        state = torch.load(root / config["lora_target"], map_location="cpu",
                           weights_only=False, mmap=True)
        result = peft.set_peft_model_state_dict(generator.model, state.get("generator_lora", state))
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise ValueError("official LongLive LoRA weights do not match the native model")
        generator.model = generator.model.merge_and_unload()
        del state
    generator.to(device=device, dtype=dtype).eval().requires_grad_(False)
    generator.model.num_frame_per_block = 3
    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval().requires_grad_(False)
    text = WanTextEncoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
    ids, mask = text.tokenizer([args.prompt], return_mask=True, add_special_tokens=True)
    length = int(mask.sum(dim=1).max())
    ids, mask = ids[:, :length].to(device), mask[:, :length].to(device)
    embeddings = text.text_encoder(ids, mask)
    embeddings[~mask.bool()] = 0
    condition = {"prompt_embeds": embeddings}
    del text
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    frame_tokens = (args.height // 16) * (args.width // 16)
    generator.seq_len = 3 * frame_tokens
    caches, cross = [], []
    for block in generator.model.blocks:
        attention_block = block.self_attn
        heads = attention_block.num_heads
        head_dim = attention_block.q.out_features // heads
        capacity = config["local_frames"] * frame_tokens
        attention_block.max_attention_size = capacity
        caches.append({"k": torch.zeros(1, capacity, heads, head_dim, device=device, dtype=dtype),
                       "v": torch.zeros(1, capacity, heads, head_dim, device=device, dtype=dtype),
                       "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
                       "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
                       "cpu_k_frames": [], "cpu_v_frames": []})
        cross.append({"k": torch.zeros(1, 512, heads, head_dim, device=device, dtype=dtype),
                      "v": torch.zeros(1, 512, heads, head_dim, device=device, dtype=dtype), "is_init": False})
    pipe = SimpleNamespace(generator=generator, vae=vae, kv_cache1=caches,
                           crossattn_cache=cross, frame_seq_length=frame_tokens)
    adapter = (SelfForcingHistoryAdapter(pipe, max_references=args.k) if backend == "self-forcing"
               else LongLiveHistoryAdapter(pipe))
    scheduler = generator.get_scheduler()
    steps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0.])))[1000 - torch.tensor([1000, 750, 500, 250])]
    consumed = []

    def history_read(module, inputs, kwargs):
        if backend == "longlive":
            import inspect
            arguments = inspect.signature(module.forward).bind_partial(*inputs, **kwargs).arguments
            memory_indices = arguments.get("memory_indices")
            if memory_indices is not None:
                consumed.append(memory_indices.shape[1] * frame_tokens)
            return
        cache = kwargs.get("kv_cache")
        if cache is None and len(inputs) > 5:
            cache = inputs[5]
        if cache is not None and cache["k"].shape[1] > config["local_frames"] * frame_tokens:
            consumed.append(cache["k"].shape[1])

    handles = [block.self_attn.register_forward_pre_hook(history_read, with_kwargs=True)
               for block in generator.model.blocks]

    def chunk(index, use_history):
        rng = torch.Generator(device=device).manual_seed(args.seed + index)
        shape = (1, 3, 16, args.height // 8, args.width // 8)
        x = torch.randn(shape, generator=rng, device=device, dtype=dtype)
        forward = adapter.forward if use_history else generator
        kwargs = {} if use_history else {"kv_cache": pipe.kv_cache1, "crossattn_cache": pipe.crossattn_cache}
        for offset, step in enumerate(steps):
            t = torch.ones((1, 3), device=device, dtype=torch.int64) * step.item()
            _, clean = forward(noisy_image_or_video=x, conditional_dict=condition, timestep=t,
                               current_start=index * 3 * frame_tokens, **kwargs)
            if offset + 1 < len(steps):
                noise = torch.randn(clean.shape, generator=rng, device=device, dtype=dtype)
                x = scheduler.add_noise(clean.flatten(0, 1), noise.flatten(0, 1),
                    steps[offset + 1] * torch.ones(3, device=device)).unflatten(0, (1, 3))
        forward(noisy_image_or_video=clean, conditional_dict=condition, timestep=torch.zeros_like(t),
                current_start=index * 3 * frame_tokens, **kwargs)
        if not torch.isfinite(clean).all():
            raise ValueError("nonfinite generated latents")
        return clean

    def decode(latents):
        vae.model.clear_cache()
        pixels = ((vae.decode_to_pixel(latents).float() + 1) / 2).clamp(0, 1)[0].cpu()
        if not torch.isfinite(pixels).all() or pixels.shape[1:] != (3, args.height, args.width):
            raise ValueError("invalid native decoded video")
        return pixels

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    selector = FrameMorrow(SelectorConfig(**checkpoint["config"])).to(device).eval()
    selector.load_state_dict(checkpoint["model"], strict=True)
    limit = (checkpoint.get("provenance", {}).get("encoder_metadata") or {}).get("max_condition_tokens", 256)
    encoders = None
    pieces, baseline, selections = [], [], []
    baseline_kv = baseline_cross = None
    differences = []
    for index in range(args.chunks):
        if index >= args.warmup_chunks:
            pixels = decode(torch.cat(pieces, dim=1))
            now = (len(pixels) - 1) / 16
            sampled = [round((.25 + .5 * i) * 16) for i in range(max(0, math.floor((now - .25) * 2) + 1))]
            history = [i for i in sampled if i / 16 <= now - 4]
            recent = [i for i in sampled if i / 16 > now - 4]
            if len(history) > 128:
                history = [history[round(i * (len(history) - 1) / 127)] for i in range(128)]
            if not history or len(recent) != 8:
                raise ValueError("warmup is too short for historical candidates and four seconds of recent context")
            frames = output / "frames"
            frames.mkdir(exist_ok=True)
            paths = []
            for frame in history + recent:
                path = frames / f"{frame:06d}.png"
                if not path.exists():
                    Image.fromarray((pixels[frame].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)
                paths.append(path)
            if encoders is None:
                encoders = FrozenEncoders("text", args.visual_model, args.condition_model,
                                          device=str(device), max_condition_tokens=limit)
                text_features = encoders.text(args.prompt)[None].to(device)
            features = encoders.images(paths)
            times = torch.tensor([[i / 16 for i in history]], dtype=torch.float64, device=device)
            selected, scores = selector.select(features[:len(history)][None].to(device),
                features[len(history):][None].to(device), text_features, times, k=args.k)
            kwargs = {'history_latent_indices': [frame // 4 for frame in history]} if backend == 'self-forcing' else {}
            selection = adapter.refresh(pixels[history], times[0].tolist(), selected[0],
                recent_start=recent[0] / 16, conditional_dict=condition, **kwargs)
            selection.update(chunk=index, frame_indices=[history[i] for i in selection["indices"]],
                             scores=scores[0, selected[0]].tolist(), candidate_count=len(history),
                             recent_start=recent[0] / 16, observed_until=now)
            selections.append(selection)
            if args.compare_baseline:
                if baseline_kv is None:
                    baseline_kv, baseline_cross = copy.deepcopy(pipe.kv_cache1), copy.deepcopy(pipe.crossattn_cache)
                live_kv, live_cross = pipe.kv_cache1, pipe.crossattn_cache
                pipe.kv_cache1, pipe.crossattn_cache = baseline_kv, baseline_cross
                reads_before = len(consumed)
                plain = chunk(index, False)
                if len(consumed) != reads_before:
                    raise RuntimeError("baseline unexpectedly consumed selected references")
                pipe.kv_cache1, pipe.crossattn_cache = live_kv, live_cross
                baseline.append(plain)
        clean = chunk(index, index >= args.warmup_chunks)
        if backend == 'self-forcing':
            adapter.capture(index * 3, clean.shape[1])
        pieces.append(clean)
        if index < args.warmup_chunks:
            baseline.append(clean)
        elif args.compare_baseline:
            differences.append((clean.float() - plain.float()).abs().max().item())
        log(f"Generated chunk {index + 1}/{args.chunks}, references={len(adapter.selected_indices)}")
    for handle in handles:
        handle.remove()
    if not consumed or (args.compare_baseline and max(differences) <= 0):
        raise RuntimeError("selected historical references were not consumed")

    def save_video(name, latents):
        pixels = decode(torch.cat(latents, dim=1))
        frames = (pixels.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", "16", "-i", "pipe:0",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(output / name)],
            input=frames.tobytes(), check=True)
        return len(frames)

    count = save_video("conditioned.mp4", pieces)
    if args.compare_baseline:
        assert save_video("baseline.mp4", baseline) == count
    (output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    report = {"backend": backend, "commit": revision, "device": str(device), "attention": "sdpa",
              "prompt": args.prompt, "seed": args.seed, "chunks": args.chunks,
              "warmup_chunks": args.warmup_chunks, "k": args.k,
              "selector_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "height": args.height, "width": args.width, "frames": count, "fps": 16,
              "reference_attention_calls": len(consumed), "baseline_max_latent_differences": differences,
              "elapsed_seconds": time.monotonic() - started, "scope": "execution only, no quality improvement claim"}
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    log(f"Saved {output / 'conditioned.mp4'}")
