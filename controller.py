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

RESTORE_MODES = [
    "smart: manual mask, local auto, or full frame",
    "full generated frame",
    "manual mask only",
    "auto local edit only",
]

ASPECTS = [
    "1:1 (Square)", "3:4 (Portrait Standard)", "4:3 (Landscape Standard)",
    "9:16 (Portrait Tall)", "16:9 (Landscape Wide)", "2:3 (Portrait Photo)",
    "3:2 (Landscape Photo)",
]


def _files(kind):
    try:
        return list(folder_paths.get_filename_list(kind))
    except Exception:
        return []


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
        return io.Schema(
            node_id="KreaAIO",
            display_name="Krea2 AIO AJ",
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
                io.Boolean.Input("remove_background", default=False,
                                 tooltip="Pipeline 1 only: RMBG-2.0 on source and reference. "
                                         "Leave off for scene-preserving edits like clothing."),

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
                io.Float.Input("megapixels", default=1.0, min=0.1, max=16.0, step=0.1,
                               tooltip="ABSOLUTE megapixel target, not a multiplier."),
                io.Int.Input("grounding_px", default=768, min=0, max=4096, step=64),
                io.Float.Input("ref_boost", default=1.0, min=0.0, max=1000.0, step=0.01),
                io.Combo.Input("restore_mode", options=RESTORE_MODES, default=RESTORE_MODES[0]),

                io.String.Input("save_root", default="Krea2AJ",
                                tooltip="Top folder under ComfyUI/output. Each pipeline "
                                        "files itself into its own subfolder automatically."),

                io.String.Input("loras_json", default="[]",
                                tooltip="LoRA stack, managed by the node UI."),

                io.Combo.Input("unet_name", options=_files("diffusion_models") or ["none"],
                               default=pipelines.KREA_UNET),
                io.Combo.Input("clip_name", options=_files("text_encoders") or ["none"],
                               default=pipelines.KREA_CLIP),
                io.Combo.Input("vae_name", options=_files("vae") or ["none"],
                               default=pipelines.KREA_VAE),

                # Pipeline 5 runs a different model family, so it needs its own loaders.
                io.Combo.Input("flux_unet_name", options=_files("diffusion_models") or ["none"],
                               default=pipelines.FLUX_UNET,
                               tooltip="Upscale only: Flux 2 Klein diffusion model."),
                io.Combo.Input("flux_clip_name", options=_files("text_encoders") or ["none"],
                               default=pipelines.FLUX_CLIP,
                               tooltip="Upscale only: Flux 2 text encoder (type flux2)."),
                io.Combo.Input("flux_vae_name", options=_files("vae") or ["none"],
                               default=pipelines.FLUX_VAE,
                               tooltip="Upscale only: Flux 2 VAE."),
                io.Int.Input("upscale_steps", default=2, min=1, max=50,
                             tooltip="Upscale only."),

                io.Int.Input("outpaint_bottom", default=256, min=0, max=2048, step=8),
                io.Int.Input("outpaint_top", default=0, min=0, max=2048, step=8),
                io.Int.Input("outpaint_left", default=0, min=0, max=2048, step=8),
                io.Int.Input("outpaint_right", default=0, min=0, max=2048, step=8),
                io.Int.Input("outpaint_feather", default=8, min=0, max=128, step=1,
                             tooltip="Softness of the pad mask edge. High values leave "
                                     "partially-green pixels in the transition band, which "
                                     "show up as a green seam after the restore."),

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
    def execute(cls, pipeline, upscale, edit_mode, fill_mode, face_detail,
                remove_background, prompt,
                seed, steps, cfg, sampler, scheduler, aspect_ratio, megapixels,
                grounding_px, ref_boost, restore_mode, save_root, loras_json,
                unet_name, clip_name, vae_name,
                flux_unet_name, flux_clip_name, flux_vae_name, upscale_steps,
                outpaint_bottom, outpaint_top, outpaint_left, outpaint_right,
                outpaint_feather,
                image=None, mask=None, reference=None) -> io.NodeOutput:

        try:
            loras = json.loads(loras_json) if loras_json else []
            if not isinstance(loras, list):
                loras = []
        except Exception:
            log.warning("[KreaAIO] loras_json is not valid JSON; ignoring it")
            loras = []

        ctx = pipelines.Ctx(
            prompt=prompt, seed=seed, steps=steps, cfg=cfg,
            sampler=sampler, scheduler=scheduler,
            aspect_ratio=aspect_ratio, megapixels=megapixels,
            grounding_px=grounding_px, ref_boost=ref_boost,
            restore_mode=restore_mode, loras=loras,
            image=image, mask=mask, reference=reference,
            remove_background=remove_background,
            upscale_megapixels=megapixels,
            mode_b=edit_mode.startswith("B"),
            outpaint=fill_mode.startswith("B"),
            face_detail=face_detail,
            unet_name=unet_name, clip_name=clip_name, vae_name=vae_name,
            flux_unet_name=flux_unet_name, flux_clip_name=flux_clip_name,
            flux_vae_name=flux_vae_name, upscale_steps=upscale_steps,
            pad_bottom=outpaint_bottom, pad_top=outpaint_top,
            pad_left=outpaint_left, pad_right=outpaint_right,
            pad_feather=outpaint_feather,
        )

        idx = PIPELINES.index(pipeline) + 1 if pipeline in PIPELINES else 4
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

        # Preview only. Saving is the job of a SaveImage node wired to the image
        # output, which is where the user picks the folder and prefix.
        save_path = build_save_path(save_root, idx, ctx.mode_b, ctx.outpaint,
                                    face_detail, upscale)

        ui = engine.preview_images(image)
        return io.NodeOutput(image, source, save_path, ui=ui)
