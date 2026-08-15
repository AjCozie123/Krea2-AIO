"""KREA2 AJ AIO — the whole 5-pipeline workflow inside one node.

The pipelines live in pipelines.py and are built by calling the real ComfyUI node
classes through engine.call(), so nothing here reimplements anyone's algorithm.

The node is an output node: it runs the selected pipeline and saves + previews the
result, so a bare graph containing only this node is a complete workflow.
"""

import json
import logging
import traceback

import folder_paths

from comfy_api.latest import io

from . import engine, models, pipelines

log = logging.getLogger(__name__)


PIPELINES = [
    "1 - CLASSIC EDIT (two references)",
    "2 - IDENTITY / OSTRIS EDIT",
    "3 - GREEN-MASK INPAINT / OUTPAINT",
    "4 - TEXT TO IMAGE",
]

EDIT_MODES = ["A - Native Krea2Edit (identity)", "B - Ostris Edit (ai-toolkit)"]
FILL_MODES = ["A - INPAINT (you paint the mask)", "B - OUTPAINT (auto green border)"]

# Sentinel for the prompt-enhancer model dropdown meaning "reuse the loaded text encoder".
ENHANCER_LOADED = "(loaded text encoder · low VRAM)"

# Preset max-token sizes for the enhancer, as a dropdown (no free typing). The pipeline reads
# the leading number; the labels just tell the user the VRAM/speed tradeoff.
ENHANCER_TOKENS = ["256 (low VRAM · fast)", "512 (medium)", "1024 (high VRAM)",
                   "2048 (highest VRAM · slowest)"]

# How pipelines 1 and 2 decide their output size. "Match input image size" keeps the edit at
# the source image's own dimensions (best identity/edit adherence); the other lets you force an
# explicit aspect + megapixel target.
SIZE_MODES = [
    "Match input image size (keeps the edit's own size)",
    "Aspect ratio + megapixels (choose a resolution)",
]

RESTORE_MODES = [
    "smart: manual mask, local auto, or full frame",
    "full generated frame",
    "manual mask only",
    "auto local edit only",
]

def _aspect_options():
    """Take the aspect list from the real ResolutionSelector node.

    These strings are dict KEYS in its ASPECT_RATIOS table, so a label that merely
    looks plausible raises KeyError deep inside sampling instead of failing
    validation. Never hand-write them.
    """
    fallback = [
        "1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)",
        "3:4 (Portrait Standard)", "4:3 (Standard)",
        "9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "21:9 (Ultrawide)",
    ]
    try:
        cls = engine.get_class("ResolutionSelector")
        for i in cls.define_schema().inputs:
            if getattr(i, "id", None) != "aspect_ratio":
                continue
            opts = [str(getattr(o, "value", o)) for o in (i.options or [])]
            if opts:
                return opts
    except Exception as e:
        log.warning("[KreaAIO] could not read ResolutionSelector aspects (%s); "
                    "using the built-in list", e)
    return fallback


ASPECTS = _aspect_options()


def _files(kind):
    try:
        return list(folder_paths.get_filename_list(kind))
    except Exception:
        return []


def _base(name):
    return str(name).replace("\\", "/").split("/")[-1].lower()


def _pick_default(options, preferred):
    """Choose a default that actually EXISTS in this install.

    The hardcoded preferred paths (pipelines.KREA_* / FLUX_*) use the author's own
    model-folder layout (subfolders + backslashes). On any other machine those exact
    strings are not in the file list, so ComfyUI fails prompt validation with
    'Value not in list' before the node even runs. This resolves to:
      1. the exact preferred path if present,
      2. otherwise any file with the SAME filename in a different folder,
      3. otherwise the first available file (always valid),
    so a fresh install never hard-fails and flat model folders auto-match.
    """
    if not options:
        return "none"
    if preferred in options:
        return preferred
    pb = _base(preferred)
    for o in options:
        if _base(o) == pb:
            return o
    return options[0]


def _images():
    try:
        return sorted(folder_paths.filter_files_content_types(
            __import__("os").listdir(folder_paths.get_input_directory()), ["image"]))
    except Exception:
        try:
            import os
            d = folder_paths.get_input_directory()
            return sorted(f for f in os.listdir(d)
                          if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")))
        except Exception:
            return []


# Per-pipeline output folders, so switching pipeline re-files the results without
# anyone editing SaveImage. Wire save_path -> SaveImage.filename_prefix.
SAVE_FOLDERS = {
    1: ("Classic", "CE"),
    2: ("Identity", "ID"),
    3: ("Inpaint", "IN"),
    4: ("TextToImage", "T2I"),
}


def build_save_path(root, idx, mode_b, outpaint, face_detail, upscale):
    root = (root or "Krea2AJ").strip().strip("/\\") or "Krea2AJ"
    folder, prefix = SAVE_FOLDERS.get(idx, ("Misc", "OUT"))

    # the sub-modes deserve their own folders too
    if idx == 2:
        folder, prefix = ("Identity/OstrisB", "OSB") if mode_b else ("Identity/ModeA", "IDA")
    elif idx == 3:
        folder, prefix = ("Outpaint", "OUT") if outpaint else ("Inpaint", "IN")

    if face_detail:
        prefix += "-fd"
    if upscale:
        folder += "/Upscaled"
        prefix += "-4K"
    return f"{root}/{folder}/{prefix}"


class KreaAIO(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        imgs = _images() or ["example.png"]
        _unet_opts = _files("diffusion_models") or ["none"]
        _clip_opts = _files("text_encoders") or ["none"]
        _vae_opts = _files("vae") or ["none"]
        # Prompt-enhancer LLM choice. The first option reuses whatever text encoder the
        # pipeline already loads (Qwen3-VL) — zero extra VRAM. Any other pick loads that
        # encoder just for the rewrite (use an abliterated Qwen3-VL here for no refusals).
        _enh_opts = [ENHANCER_LOADED] + _clip_opts
        return io.Schema(
            node_id="KreaAIO",
            display_name="Krea2 AIO",
            category="KREA2",
            description=(
                "The whole 5-pipeline KREA2 workflow in one node. Pick a pipeline and the "
                "node runs it end to end: models, LoRAs, encoders, reference method, "
                "sampler, restore and save."
            ),
            is_output_node=True,
            inputs=[
                io.Combo.Input("pipeline", options=PIPELINES, default=PIPELINES[3]),
                io.Boolean.Input("upscale", default=False,
                                 tooltip="Run the FLUX 2 Klein detail upscale on the result."),
                io.Combo.Input("edit_mode", options=EDIT_MODES, default=EDIT_MODES[0]),
                io.Combo.Input("fill_mode", options=FILL_MODES, default=FILL_MODES[0]),
                io.Boolean.Input("face_detail", default=False),

                # Images come from real LoadImage nodes outside this one. That keeps
                # ComfyUI's MaskEditor working normally: paint the alpha mask on the
                # LoadImage node and wire its MASK output to `mask` below.
                io.Image.Input("image", optional=True,
                               tooltip="Source image. Required by pipelines 1, 2 and 3."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Only used by pipeline 3 INPAINT. The alpha mask "
                                      "painted in MaskEditor — never green pixels."),
                io.Image.Input("reference", optional=True,
                               tooltip="Second reference (subject / donor). Training order "
                                       "is scene first, subject second."),
                io.Boolean.Input("use_reference", default=True,
                                 tooltip="Ignore the reference socket without unwiring it. "
                                         "Off = single-reference edit."),
                io.Boolean.Input("remove_background", default=True,
                                 tooltip="Pipeline 1 only: RMBG-2.0 on source and reference, "
                                         "as the reference workflow does. Turn OFF for "
                                         "scene-preserving edits such as a clothing change."),

                io.String.Input("prompt", multiline=True, default=""),

                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True),
                io.Int.Input("steps", default=8, min=1, max=200),
                io.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1),
                io.Combo.Input("sampler", options=["euler", "er_sde", "dpmpp_2m", "ddim", "lcm"],
                               default="euler"),
                io.Combo.Input("scheduler", options=["simple", "beta", "normal", "karras", "sgm_uniform"],
                               default="simple"),

                io.Combo.Input("aspect_ratio", options=ASPECTS, default=ASPECTS[1]),
                io.Float.Input("megapixels", default=2.0, min=0.1, max=16.0, step=0.1,
                               tooltip="ABSOLUTE megapixel target, not a multiplier."),
                io.Int.Input("grounding_px", default=768, min=0, max=4096, step=64),
                io.Float.Input("ref_boost", default=1.0, min=0.0, max=1000.0, step=0.01),
                io.Combo.Input("restore_mode", options=RESTORE_MODES, default=RESTORE_MODES[0]),

                io.String.Input("save_root", default="Krea2AJ",
                                tooltip="Save name/path under ComfyUI/output — YOU decide it. "
                                        "Used verbatim as SaveImage's filename_prefix, so it can "
                                        "include folders and a name, e.g. 'MyProject/portrait'. "
                                        "Turn on auto_organize to instead file each pipeline into "
                                        "its own subfolder automatically."),

                io.String.Input("loras_json", default="[]",
                                tooltip="LoRA stack, managed by the node UI."),

                io.Combo.Input("unet_name", options=_unet_opts,
                               default=_pick_default(_unet_opts, pipelines.KREA_UNET)),
                io.Combo.Input("clip_name", options=_clip_opts,
                               default=_pick_default(_clip_opts, pipelines.KREA_CLIP)),
                io.Combo.Input("vae_name", options=_vae_opts,
                               default=_pick_default(_vae_opts, pipelines.KREA_VAE)),

                # Pipeline 5 runs a different model family, so it needs its own loaders.
                io.Combo.Input("flux_unet_name", options=_unet_opts,
                               default=_pick_default(_unet_opts, pipelines.FLUX_UNET),
                               tooltip="Upscale only: Flux 2 Klein diffusion model."),
                io.Combo.Input("flux_clip_name", options=_clip_opts,
                               default=_pick_default(_clip_opts, pipelines.FLUX_CLIP),
                               tooltip="Upscale only: Flux 2 text encoder (type flux2)."),
                io.Combo.Input("flux_vae_name", options=_vae_opts,
                               default=_pick_default(_vae_opts, pipelines.FLUX_VAE),
                               tooltip="Upscale only: Flux 2 VAE."),
                io.Int.Input("upscale_steps", default=2, min=1, max=50,
                             tooltip="Upscale only."),
                io.Float.Input("upscale_megapixels", default=4.0, min=0.5, max=16.0, step=0.5,
                               tooltip="Upscale only: ABSOLUTE megapixel target, as the "
                                       "reference workflow uses (4.0). Lower it on 8 GB."),

                io.Int.Input("outpaint_bottom", default=256, min=0, max=2048, step=8),
                io.Int.Input("outpaint_top", default=0, min=0, max=2048, step=8),
                io.Int.Input("outpaint_left", default=0, min=0, max=2048, step=8),
                io.Int.Input("outpaint_right", default=0, min=0, max=2048, step=8),
                io.Int.Input("outpaint_feather", default=0, min=0, max=128, step=1,
                             tooltip="Softness of the pad mask edge. The reference workflow "
                                     "uses 24, but that leaves partially-green pixels in the "
                                     "transition band which the restore blends back as a "
                                     "green seam — measured 3559 green pixels at 24 vs 0 at "
                                     "0. Left at 0 deliberately."),

                # Appended AT THE END so existing saved workflows keep their widget
                # positions. denoise feeds every pipeline's KSampler; 1.0 = current
                # behaviour. state_json is the per-pipeline UI store, managed by the JS.
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="KSampler denoise. 1.0 = full generation (default). "
                                       "Lower it for a lighter, more faithful pass."),
                io.String.Input("state_json", default="{}",
                                tooltip="Per-pipeline settings, prompts and LoRA stacks, "
                                        "managed by the node UI. Do not edit by hand."),
                # OFF by default: the save name/path is used exactly as you type it, so YOU
                # decide the folder and filename. ON restores the automatic per-pipeline
                # subfolder organisation. Appended last to keep widget positions stable.
                io.Boolean.Input("auto_organize", default=False,
                                 tooltip="Off (default): save name/path is used exactly as you "
                                         "type it. On: auto-file each pipeline into its own "
                                         "subfolder under the name you gave."),
                # One field for ALL LoRA trigger words — appended to the prompt for you.
                io.String.Input("trigger_words", default="", multiline=True,
                                tooltip="All your LoRA trigger words in one place; appended to "
                                        "the prompt automatically so you don't type them there."),

                # LLM prompt enhancer — the same idea as ComfyUI's official Krea-2 workflow:
                # rewrite your prompt with core TextGenerate on a Qwen3-VL text encoder, using
                # a per-pipeline system prompt (see prompt_enhancer.py). OFF by default (the
                # enable/disable tick). Appended last so saved workflows keep widget positions.
                io.Boolean.Input("enhance_prompt", default=False,
                                 tooltip="Rewrite your prompt with an LLM before generating "
                                         "(per-workflow system prompt). Trigger words are kept "
                                         "verbatim and appended AFTER enhancement."),
                io.Combo.Input("enhancer_model", options=_enh_opts, default=_enh_opts[0],
                               tooltip="Which LLM rewrites the prompt. '(loaded text encoder)' "
                                       "reuses the Qwen3-VL you already load — no extra VRAM. "
                                       "Pick an abliterated Qwen3-VL here to avoid refusals."),
                io.Combo.Input("llm_max_token", options=ENHANCER_TOKENS, default=ENHANCER_TOKENS[0],
                               tooltip="Length cap for the enhanced prompt. 256 = low VRAM / fast; "
                                       "higher = longer, richer prompts but more VRAM and slower. "
                                       "The prompt is front-loaded so it stays complete at any size."),

                # Pipelines 1 & 2 only: keep the edit at the source image's own size, or force an
                # explicit aspect + megapixel target. Appended last to keep widget positions stable.
                # Negative prompt. Krea 2 TURBO runs at cfg 1.0, where classifier-free guidance
                # is off and a negative is mathematically inert — so this only does anything with
                # a RAW checkpoint at cfg > 1 (~5.5, ~28 steps). Below cfg 1.05 we keep the old
                # zero-out behaviour so turbo is unchanged. The UI says the same thing.
                io.String.Input("negative_prompt", default="", multiline=True,
                                tooltip="Only works when CFG is above 1.0 (i.e. a RAW checkpoint). "
                                        "Krea 2 Turbo runs at CFG 1.0 where negatives do nothing. "
                                        "Useful for things the model adds unasked, e.g. "
                                        "'shallow depth of field, bokeh, blurred foreground'."),
                io.Combo.Input("size_mode", options=SIZE_MODES, default=SIZE_MODES[0],
                               tooltip="Classic / Identity output size. 'Match input image size' "
                                       "keeps the edit at the source image's own dimensions — best "
                                       "for identity and edit adherence. 'Aspect ratio + megapixels' "
                                       "forces the resolution you pick instead (can crop framing)."),

            ],
            outputs=[
                io.Image.Output(display_name="image"),
                # The original, at native resolution, so an Image Comparer can do
                # before/after without a second LoadImage.
                io.Image.Output(display_name="source"),
                # Wire this into SaveImage.filename_prefix and every pipeline files
                # itself into its own folder with no typing.
                io.String.Output(display_name="save_path"),
            ],
        )

    @classmethod
    def validate_inputs(cls, unet_name=None, clip_name=None, vae_name=None,
                        flux_unet_name=None, flux_clip_name=None, flux_vae_name=None):
        """Skip ComfyUI's 'Value not in list' pre-check on the model dropdowns.

        ComfyUI validates EVERY input on a node at queue time, regardless of whether
        the code path uses it. So a stale or foreign model path — most often the three
        FLUX loaders, which only run for the optional upscale layer — would block the
        whole node from generating even with the upscaler off.

        Naming these inputs here defers their validation to us; returning True lets the
        run proceed. The node loads each model lazily and only when a pipeline actually
        needs it, so a genuinely missing model raises a clear error at load time instead
        of a cryptic pre-run block. Every other input keeps its normal validation.
        """
        return True

    @classmethod
    def execute(cls, pipeline, upscale, edit_mode, fill_mode, face_detail,
                use_reference, remove_background, prompt,
                seed, steps, cfg, sampler, scheduler, aspect_ratio, megapixels,
                grounding_px, ref_boost, restore_mode, save_root, loras_json,
                unet_name, clip_name, vae_name,
                flux_unet_name, flux_clip_name, flux_vae_name, upscale_steps,
                upscale_megapixels,
                outpaint_bottom, outpaint_top, outpaint_left, outpaint_right,
                outpaint_feather,
                denoise=1.0, state_json="{}", auto_organize=False, trigger_words="",
                enhance_prompt=False, enhancer_model=None, llm_max_token=256,
                size_mode=None, negative_prompt="",
                image=None, mask=None, reference=None) -> io.NodeOutput:

        try:
            loras = json.loads(loras_json) if loras_json else []
            if not isinstance(loras, list):
                loras = []
        except Exception:
            log.warning("[KreaAIO] loras_json is not valid JSON; ignoring it")
            loras = []

        # Pipeline index (1-4) — needed now so the enhancer can pick the right per-workflow
        # system prompt. The prompt is finalised INSIDE the pipeline, after the text encoder
        # loads: LLM-enhanced first (if enabled), then trigger words appended verbatim.
        idx = PIPELINES.index(pipeline) + 1 if pipeline in PIPELINES else 4
        tw = (trigger_words or "").strip()

        ctx = pipelines.Ctx(
            prompt=prompt, seed=seed, steps=steps, cfg=cfg, denoise=denoise,
            sampler=sampler, scheduler=scheduler,
            aspect_ratio=aspect_ratio, megapixels=megapixels,
            grounding_px=grounding_px, ref_boost=ref_boost,
            restore_mode=restore_mode, loras=loras,
            image=image, mask=mask,
            # honour the toggle rather than making the user unwire the LoadImage
            reference=(reference if use_reference else None),
            remove_background=remove_background,
            mode_b=edit_mode.startswith("B"),
            outpaint=fill_mode.startswith("B"),
            face_detail=face_detail,
            unet_name=unet_name, clip_name=clip_name, vae_name=vae_name,
            flux_unet_name=flux_unet_name, flux_clip_name=flux_clip_name,
            flux_vae_name=flux_vae_name, upscale_steps=upscale_steps,
            upscale_megapixels=upscale_megapixels,
            pad_bottom=outpaint_bottom, pad_top=outpaint_top,
            pad_left=outpaint_left, pad_right=outpaint_right,
            pad_feather=outpaint_feather,
            # prompt finalisation (LLM enhancer + trigger words), applied in the pipeline
            enhance_prompt=enhance_prompt, enhancer_model=enhancer_model,
            llm_max_token=llm_max_token, trigger_words=tw, pipeline_idx=idx,
            size_mode=size_mode, negative_prompt=(negative_prompt or "").strip(),
        )
        RUNNERS = {
            1: pipelines.classic_edit,
            2: pipelines.identity_edit,
            3: pipelines.green_mask_fill,
            4: pipelines.text_to_image,
        }
        runner = RUNNERS.get(idx)
        if runner is None:
            raise engine.NodeCallError(f"unknown pipeline {idx}")

        try:
            image, source = runner(ctx)
            if upscale:
                image = pipelines.klein_upscale(ctx, image)
        except engine.NodeCallError:
            raise
        except Exception as e:
            log.error("[KreaAIO] pipeline %s failed:\n%s", idx, traceback.format_exc())
            raise RuntimeError(f"KREA2 AIO pipeline {idx} failed: {e}") from e

        # By default the user's typed name/path is used verbatim (they decide folder +
        # filename). auto_organize restores the automatic per-pipeline subfoldering.
        if auto_organize:
            save_path = build_save_path(save_root, idx, ctx.mode_b, ctx.outpaint,
                                        face_detail, upscale)
        else:
            save_path = (save_root or "Krea2AJ").strip().strip("/\\") or "Krea2AJ"

        # Deliberately no ui IMAGES. Attaching an image preview makes ComfyUI lay an image
        # out inside this node and refit the node around it, which squashes the embedded UI
        # to a narrow strip mid-run. Results are viewed through the SaveImage / Image
        # Comparer nodes wired to the outputs.
        #
        # We DO return a tiny non-image ui payload: that is what makes ComfyUI call the
        # frontend's node.onExecuted() hook, which is how the finish chime fires (the same
        # mechanism pysssss's PlaySound uses). A custom key never triggers the image layout.
        return io.NodeOutput(image, source, save_path,
                             ui={"kaio_done": [{"pipeline": idx, "upscaled": bool(upscale)}]})
