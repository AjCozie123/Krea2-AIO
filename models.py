"""Cached model loading for the AIO node.

In the graph, UNETLoader / CLIPLoader / VAELoader are separate nodes, so ComfyUI's
execution cache keeps them loaded between runs. Inside a single node there is no such
cache, and reloading a checkpoint from disk on every queue would be brutal on an 8 GB
card. So we keep our own cache keyed on the loader arguments.

Holding a reference only prevents a re-read from disk — ComfyUI's model management
still moves the weights on and off the GPU as it sees fit, exactly as it does for the
loader nodes.
"""

import logging

from . import engine

log = logging.getLogger(__name__)

_CACHE = {}


def _cached(key, build):
    if key not in _CACHE:
        log.info("[KreaAIO] loading %s", key)
        _CACHE[key] = build()
    return _CACHE[key]


def clear_cache():
    _CACHE.clear()


def unet(unet_name, weight_dtype="default"):
    return _cached(("unet", unet_name, weight_dtype),
                   lambda: engine.call1("UNETLoader", unet_name=unet_name, weight_dtype=weight_dtype))


def clip(clip_name, clip_type, device="default"):
    def build():
        try:
            return engine.call1("CLIPLoader", clip_name=clip_name, type=clip_type, device=device)
        except Exception:
            # older signature without `device`
            return engine.call1("CLIPLoader", clip_name=clip_name, type=clip_type)
    return _cached(("clip", clip_name, clip_type, device), build)


def vae(vae_name):
    return _cached(("vae", vae_name),
                   lambda: engine.call1("VAELoader", vae_name=vae_name))


def _resolve_lora(name):
    """Map a LoRA name to a file that exists here. Handles a different folder layout
    (e.g. the default 'Krea2\\file' vs a flat 'file') by matching on filename. Returns
    None if no such LoRA is installed, so the caller can skip it instead of crashing."""
    import folder_paths
    try:
        avail = list(folder_paths.get_filename_list("loras"))   # ComfyUI caches this
    except Exception:
        avail = []
    if name in avail:
        return name
    base = str(name).replace("\\", "/").split("/")[-1].lower()
    for o in avail:
        if str(o).replace("\\", "/").split("/")[-1].lower() == base:
            return o
    return None


def apply_loras(model, clip_obj, loras):
    """Apply a LoRA stack.

    `loras` is a list of {"on": bool, "lora": name, "strength": float}. Entries are
    applied in order, matching how Power Lora Loader stacks them. LoraLoader clones the
    patcher, so the cached base model is never mutated. A LoRA that isn't installed is
    SKIPPED with a warning (never a hard crash) — the pipeline still runs without it.
    """
    for entry in loras or []:
        if not entry.get("on"):
            continue
        name = entry.get("lora")
        if not name or name == "None":
            continue
        strength = float(entry.get("strength", 1.0))
        if strength == 0.0:
            continue
        real = _resolve_lora(name)
        if real is None:
            log.warning("[KreaAIO] LoRA %r is not in your loras folder — skipping it. "
                        "Download it (see the node's 'Optional extras & LoRAs' panel) or "
                        "remove it from the stack.", name)
            continue
        name = real
        if clip_obj is not None:
            model, clip_obj = engine.call("LoraLoader", model=model, clip=clip_obj,
                                          lora_name=name, strength_model=strength,
                                          strength_clip=strength)[:2]
        else:
            model = engine.call1("LoraLoaderModelOnly", model=model,
                                 lora_name=name, strength_model=strength)
    return model, clip_obj
