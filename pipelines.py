"""The 5 KREA2 pipelines, rebuilt as Python orchestration.

Every step calls the real node class through engine.call(), so this is a wiring layer,
not a reimplementation of anyone's algorithm.

Measured facts encoded here — do not "simplify" these away:

* Krea 2 has NO default reference method. TextEncodeKrea2OstrisEdit on its own has its
  reference latents IGNORED (measured 0.11 correlation to source vs 0.997 with a method
  applied) and it fails SILENTLY. MODE A / pipeline 1 use Krea2EditModelPatch; MODE B
  and pipeline 3 use FluxKontextMultiReferenceLatentMethod("index_timestep_zero").
  One, never both.
* LoRA must match the encoder: MODE A -> krea2_identity_edit_v1_2, MODE B -> ai-toolkit.
* The inpaint mask is the source image's ALPHA channel, never green pixels.
  KreaAspectPreservePrepare paints the green region itself from that mask.
* ImageScaleToTotalPixels takes an ABSOLUTE megapixel target, not a multiplier.

Two easily-confused prepare nodes, verified against object_info:
    KreaImageAspectPreservePrepare -> 6 outputs, restore_map at index 1
    KreaAspectPreservePrepare      -> 7 outputs, restore_map at index 2
"""

import logging

from . import engine, models, prompt_enhancer

log = logging.getLogger(__name__)

KREA_UNET = "Krea2\\pornmasterKrea2_v2TurboInt8.safetensors"
KREA_CLIP = "Qwen\\qwen3vl_4b_fp8_scaled.safetensors"
KREA_CLIP_TYPE = "krea2"
KREA_VAE = "wan\\wan_2.1_vae.safetensors"

FLUX_UNET = "Flux 2 9B\\flux-2-klein-9b-fp8.safetensors"
FLUX_CLIP = "Qwen\\qwen_3_8b_fp8mixed.safetensors"
FLUX_CLIP_TYPE = "flux2"
FLUX_VAE = "Flux 2 9b\\full_encoder_small_decoder.safetensors"

SAM_MODEL = "sam_vit_b_01ec64.pth"
FACE_DETECTOR = "bbox/yolov12l-face.pt"

GREEN = 65280  # 0x00FF00


class Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get(self, k, d=None):
        return self.__dict__.get(k, d)


class PipelineInputError(RuntimeError):
    pass



# Labels this node shipped before they were sourced from ResolutionSelector. Older
# saved workflows still contain them, and they are dict KEYS in that node, so an
# unmapped one raises KeyError mid-sample. Remap instead of making the user re-pick.
LEGACY_ASPECTS = {
    "16:9 (Landscape Wide)": "16:9 (Widescreen)",
    "4:3 (Landscape Standard)": "4:3 (Standard)",
    "9:16 (Portrait Tall)": "9:16 (Portrait Widescreen)",
    "3:2 (Landscape Photo)": "3:2 (Photo)",
}


def resolve_aspect(value):
    value = value or "3:4 (Portrait Standard)"
    value = LEGACY_ASPECTS.get(value, value)
    try:
        cls = engine.get_class("ResolutionSelector")
        for i in cls.define_schema().inputs:
            if getattr(i, "id", None) == "aspect_ratio":
                valid = [str(getattr(o, "value", o)) for o in (i.options or [])]
                if valid and value not in valid:
                    raise PipelineInputError(
                        f"aspect_ratio {value!r} is not one of ResolutionSelector's "
                        f"options: {', '.join(valid)}"
                    )
    except PipelineInputError:
        raise
    except Exception:
        pass
    return value


def krea_base(ctx):
    model = models.unet(ctx.get("unet_name") or KREA_UNET)
    clip = models.clip(ctx.get("clip_name") or KREA_CLIP, KREA_CLIP_TYPE)
    vae = models.vae(ctx.get("vae_name") or KREA_VAE)
    return model, clip, vae


def finalize_prompt(ctx, clip):
    """Turn the user's raw prompt into the final prompt the pipeline samples with.

    Two steps, in order: (1) LLM enhancement, if the user ticked it — rewritten with the
    per-pipeline system prompt in prompt_enhancer.py; (2) trigger words appended verbatim,
    AFTER enhancement, so the LLM can never drop or reword a LoRA's activation tokens.
    Mutates and returns ctx.prompt. Never raises: enhancement failures fall back to the raw
    prompt (see prompt_enhancer.enhance).
    """
    base = (ctx.get("prompt") or "").strip()
    if ctx.get("enhance_prompt"):
        idx = int(ctx.get("pipeline_idx", 4) or 4)
        enh_clip = clip
        sel = ctx.get("enhancer_model")
        # A specific text-encoder pick (not the "(loaded …)" sentinel) is loaded just for the
        # rewrite — e.g. an abliterated Qwen3-VL that won't refuse spicy prompts.
        if sel and not str(sel).startswith("("):
            try:
                enh_clip = models.clip(sel, KREA_CLIP_TYPE)
            except Exception as e:
                log.warning("[KreaAIO] enhancer model %r failed to load (%s); using the loaded "
                            "text encoder for the rewrite instead.", sel, e)
                enh_clip = clip
        # Vision: show the enhancer the image(s) for edit workflows (1/2/3) so it grounds the
        # prompt in what is actually there. Text-to-image (4) has no source image. Reference 2 is
        # already None when the user disabled it (controller passes reference=None), so a disabled
        # reference is ignored automatically. Two images batch as image 1 (source) + image 2 (ref).
        vis_image = None
        if idx != 4:
            src = ctx.get("image")
            ref = ctx.get("reference")
            if src is not None and ref is not None:
                try:
                    vis_image = engine.call1("ImageBatch", image1=src, image2=ref)
                except Exception as e:
                    log.warning("[KreaAIO] could not batch source+reference for the enhancer (%s); "
                                "showing the source only.", e)
                    vis_image = src
            elif src is not None:
                vis_image = src
        base = prompt_enhancer.enhance(ctx, enh_clip, idx, base, image=vis_image)
    tw = (ctx.get("trigger_words") or "").strip()
    if tw:
        base = f"{base}, {tw}" if base else tw
    ctx.prompt = base
    return base


def negative_cond(ctx, positive, encode):
    """Build the negative conditioning.

    Krea 2 TURBO samples at cfg 1.0, where classifier-free guidance is disabled and the
    negative branch is never subtracted — a negative prompt there is mathematically inert,
    which is why "no bokeh"-style prompts appear to do nothing. So we only encode a real
    negative when cfg is meaningfully above 1 (i.e. a RAW checkpoint); otherwise we keep the
    original zero-out. `encode` is a callable that encodes a text string with the right
    encoder for this pipeline, so each pipeline stays on its own conditioning path.
    """
    neg = (ctx.get("negative_prompt") or "").strip()
    if neg and float(ctx.get("cfg", 1.0) or 1.0) > 1.05:
        try:
            return encode(neg)
        except Exception as e:
            log.warning("[KreaAIO] negative prompt failed to encode (%s); falling back to "
                        "an empty negative.", e)
    elif neg:
        log.info("[KreaAIO] a negative prompt is set but CFG is %.2f — Krea 2 Turbo runs at "
                 "cfg 1.0 where negatives do nothing. Use a RAW checkpoint at cfg ~5.5 with "
                 "~28 steps for it to take effect.", float(ctx.get("cfg", 1.0) or 1.0))
    return engine.call1("ConditioningZeroOut", conditioning=positive)


def use_resolution_selector(ctx):
    """True when the user asked for an explicit Aspect + Megapixels target (pipelines 1 & 2).

    Default is FALSE — i.e. match the input image's own size, which is what Krea2 edit is tuned
    for and what preserves identity/edit adherence best. An old saved workflow has no size_mode
    at all; treating that as 'match input' keeps those graphs behaving as they always did.
    """
    return str(ctx.get("size_mode") or "").lower().startswith("aspect")


def source_size(image, multiple=8):
    """The source image's own pixel size, floored to a multiple the sampler can use."""
    w, h = engine.call("GetImageSize", image=image)[:2]
    w = max(multiple, (int(w) // multiple) * multiple)
    h = max(multiple, (int(h) // multiple) * multiple)
    return w, h


def need_image(ctx, what="image"):
    img = ctx.get(what)
    if img is None:
        raise PipelineInputError(
            f"This pipeline needs an {what.upper()} input. Connect a LoadImage node to "
            f"the node's `{what}` socket."
        )
    return img


def face_detail(ctx, image, model, clip, vae, negative, sampler="euler"):
    fd_model = engine.call1("ModelSamplingAuraFlow", model=model, shift=0.7)
    positive = engine.call1("CLIPTextEncode", clip=clip,
                            text=ctx.get("face_prompt")
                            or "enhance the face and skin details of the subject")
    bbox = engine.call("UltralyticsDetectorProvider", model_name=FACE_DETECTOR)[0]
    sam = engine.call1("SAMLoader", model_name=SAM_MODEL, device_mode="AUTO")
    return engine.call(
        "FaceDetailer",
        image=image, model=fd_model, clip=clip, vae=vae,
        positive=positive, negative=negative, bbox_detector=bbox, sam_model_opt=sam,
        guide_size=1024, guide_size_for=True, max_size=1024,
        seed=int(ctx.seed), steps=4, cfg=1.0, sampler_name=sampler, scheduler="simple",
        denoise=0.2, feather=5, noise_mask=True, force_inpaint=True,
        bbox_threshold=0.5, bbox_dilation=10, bbox_crop_factor=3,
        sam_detection_hint="center-1", sam_dilation=0, sam_threshold=0.93,
        sam_bbox_expansion=0, sam_mask_hint_threshold=0.7,
        sam_mask_hint_use_negative="False",
        drop_size=10, wildcard="", cycle=1, inpaint_model=False,
        noise_mask_feather=20, scheduler_func_opt=None,
    )[0]


def rembg(image):
    """easy imageRemBg -> (image, mask). 'Hide' keeps it from writing preview files."""
    return engine.call("easy imageRemBg", images=image, rem_mode="RMBG-2.0",
                       image_output="Hide", save_prefix="ComfyUI",
                       torchscript_jit=False, add_background="none",
                       refine_foreground=False)[0]


# ---------------------------------------------------------------------------
# PIPELINE 4 — TEXT TO IMAGE
# ---------------------------------------------------------------------------

def text_to_image(ctx):
    model, clip, vae = krea_base(ctx)
    model, clip = models.apply_loras(model, clip, ctx.get("loras"))
    finalize_prompt(ctx, clip)

    positive = engine.call1("CLIPTextEncode", clip=clip, text=ctx.prompt)
    negative = negative_cond(ctx, positive,
                             lambda t: engine.call1("CLIPTextEncode", clip=clip, text=t))

    width, height = engine.call("ResolutionSelector",
                               aspect_ratio=resolve_aspect(ctx.get("aspect_ratio")),
                               megapixels=float(ctx.megapixels), multiple=8)[:2]
    latent = engine.call1("EmptyLatentImage", width=width, height=height, batch_size=1)

    sampled = engine.call1("KSampler", model=model, seed=int(ctx.seed), steps=int(ctx.steps),
                           cfg=float(ctx.cfg), sampler_name=ctx.get("sampler") or "euler",
                           scheduler=ctx.get("scheduler") or "simple",
                           positive=positive, negative=negative,
                           latent_image=latent, denoise=float(ctx.get("denoise", 1.0)))
    base_image = engine.call1("VAEDecode", samples=sampled, vae=vae)
    final = base_image
    if ctx.get("face_detail"):
        if engine.has("FaceDetailer"):
            final = face_detail(ctx, base_image, model, clip, vae, negative)
        else:
            log.warning("[KreaAIO] face_detail is on but FaceDetailer (ComfyUI-Impact-Pack) "
                        "isn't installed — skipping the face pass.")
    # Before/after: `source` is the image BEFORE the face-detail pass, so an Image Comparer
    # shows exactly what face detail changed. With face detail OFF there is no earlier stage
    # for text-to-image, so both outputs are the same image (nothing to compare).
    return final, base_image


# ---------------------------------------------------------------------------
# PIPELINE 1 — CLASSIC EDIT (two references)
# ---------------------------------------------------------------------------

def classic_edit(ctx):
    source = need_image(ctx, "image")
    model, clip, vae = krea_base(ctx)
    model, _ = models.apply_loras(model, None, ctx.get("loras"))
    finalize_prompt(ctx, clip)

    # scene / source: optional background removal, then an ABSOLUTE megapixel target.
    # RMBG needs ComfyUI-Easy-Use; if it isn't installed, skip it instead of erroring.
    do_rembg = bool(ctx.get("remove_background")) and engine.has("easy imageRemBg")
    if ctx.get("remove_background") and not do_rembg:
        log.warning("[KreaAIO] remove_background is on but 'easy imageRemBg' "
                    "(ComfyUI-Easy-Use) isn't installed — skipping background removal.")
    src = rembg(source) if do_rembg else source
    src = engine.call1("ImageScaleToTotalPixels", image=src, upscale_method="lanczos",
                       megapixels=float(ctx.megapixels), resolution_steps=64)
    src_latent = engine.call1("VAEEncode", pixels=src, vae=vae)

    ref = ctx.get("reference")
    ref_img = ref_latent = None
    if ref is not None:
        r = rembg(ref) if do_rembg else ref
        ref_img = engine.call1("ImageScaleToTotalPixels", image=r, upscale_method="lanczos",
                               megapixels=1.4, resolution_steps=64)
        ref_latent = engine.call1("VAEEncode", pixels=ref_img, vae=vae)

    # OUTPUT SIZE: either the source image's own dimensions (default — keeps the edit at the
    # size/aspect you fed in) or an explicit Aspect + Megapixels target that you chose.
    if use_resolution_selector(ctx):
        width, height = engine.call("ResolutionSelector",
                                    aspect_ratio=resolve_aspect(ctx.get("aspect_ratio")),
                                    megapixels=float(ctx.megapixels), multiple=8)[:2]
    else:
        width, height = source_size(src)
    target = engine.call1("EmptyHunyuanLatentVideo", width=width, height=height,
                          length=1, batch_size=1)

    # Reference method. Vision blocks in training order: scene first, subject second.
    patch_kw = dict(model=model, source_latent=src_latent, vae=vae, source_image=src,
                    target_latent=target, ref_boost=float(ctx.ref_boost),
                    ref_boost_a=1.0, fit_mode="fit")
    if ref_img is not None:
        patch_kw["source_image_b"] = ref_img
        patch_kw["source_latent_b"] = ref_latent
    patched = engine.call1("Krea2EditModelPatch",
                           **engine.only_supported("Krea2EditModelPatch", patch_kw))

    enc_kw = dict(clip=clip, image=src, prompt=ctx.prompt, grounding_px=int(ctx.grounding_px))
    if ref_img is not None:
        enc_kw["image_b"] = ref_img
    positive = engine.call1("Krea2EditGroundedEncode", **enc_kw)
    negative = negative_cond(ctx, positive, lambda t: engine.call1(
        "Krea2EditGroundedEncode", **{**enc_kw, "prompt": t}))

    # the graph shifts sampling before the sampler, and again (0.7) for face detail
    sampling = engine.call1("ModelSamplingAuraFlow", model=patched, shift=4.0)
    sampled = engine.call1("KSampler", model=sampling, seed=int(ctx.seed),
                           steps=int(ctx.steps), cfg=float(ctx.cfg),
                           sampler_name=ctx.get("sampler") or "euler",
                           scheduler=ctx.get("scheduler") or "simple",
                           positive=positive, negative=negative,
                           latent_image=target, denoise=float(ctx.get("denoise", 1.0)))
    base_image = engine.call1("VAEDecode", samples=sampled, vae=vae)
    final = base_image
    before = source  # default: compare the finished edit against the original input
    if ctx.get("face_detail"):
        if engine.has("FaceDetailer"):
            final = face_detail(ctx, base_image, patched, clip, vae, negative)
            before = base_image   # face detail on: compare pre/post face detail instead
        else:
            log.warning("[KreaAIO] face_detail is on but FaceDetailer (ComfyUI-Impact-Pack) "
                        "isn't installed — skipping the face pass.")
    return final, before


# ---------------------------------------------------------------------------
# PIPELINE 2 — IDENTITY / OSTRIS EDIT
# ---------------------------------------------------------------------------

def identity_edit(ctx):
    source = need_image(ctx, "image")
    model, clip, vae = krea_base(ctx)
    model, _ = models.apply_loras(model, None, ctx.get("loras"))
    finalize_prompt(ctx, clip)

    # 6 outputs: prepared_image, restore_map, width, height, batch_size, prepare_info
    prep = engine.call("KreaImageAspectPreservePrepare", image=source)
    prepared, restore_map, width, height = prep[0], prep[1], prep[2], prep[3]

    # OUTPUT SIZE. By DEFAULT identity keeps the width/height the prepare node derived from the
    # SOURCE image above — that is what Krea2 edit is tuned for and it preserves identity and edit
    # adherence best. Choosing "Aspect ratio + megapixels" instead forces your own target: it works,
    # but a different aspect crops the framing and can soften adherence. Either way the restore step
    # below re-composites onto the original. Falls back to the source size if the selector errors.
    if use_resolution_selector(ctx):
        try:
            width, height = engine.call("ResolutionSelector",
                                        aspect_ratio=resolve_aspect(ctx.get("aspect_ratio")),
                                        megapixels=float(ctx.get("megapixels") or 1.0), multiple=8)[:2]
        except Exception as e:
            log.warning("[KreaAIO] resolution selector failed for identity (%s); using the "
                        "source-derived size instead.", e)

    # The source goes in through the REFERENCE path, not the latent. The sampler and
    # the patch's target_latent both take a CLEAN Wan21-format empty latent at the
    # target size — feeding them the VAE-encoded source instead misaligns the
    # reference geometry against the output grid and wrecks edit adherence.
    source_latent = engine.call1("VAEEncode", pixels=prepared, vae=vae)
    target = engine.call1("EmptyHunyuanLatentVideo", width=width, height=height,
                          length=1, batch_size=1)

    ref = ctx.get("reference")

    if ctx.get("mode_b"):
        # MODE B — Ostris. The reference method is REQUIRED and must not be combined
        # with Krea2EditModelPatch.
        kw = dict(clip=clip, vae=vae, image1=prepared, prompt=ctx.prompt)
        if ref is not None:
            kw["image2"] = ref
        positive = engine.call1("TextEncodeKrea2OstrisEdit", **kw)
        _neg_txt = (ctx.get("negative_prompt") or "").strip() \
            if float(ctx.get("cfg", 1.0) or 1.0) > 1.05 else ""
        negative = engine.call1("TextEncodeKrea2OstrisEdit", clip=clip, vae=vae,
                                image1=prepared, prompt=_neg_txt)
        positive = engine.call1("FluxKontextMultiReferenceLatentMethod",
                                conditioning=positive,
                                reference_latents_method="index_timestep_zero")
        negative = engine.call1("FluxKontextMultiReferenceLatentMethod",
                                conditioning=negative,
                                reference_latents_method="index_timestep_zero")
        sampling = model
    else:
        # MODE A — native Krea2Edit.
        positive = engine.call1("Krea2EditGroundedEncode", clip=clip, image=prepared,
                                prompt=ctx.prompt, grounding_px=int(ctx.grounding_px))
        _neg_txt = (ctx.get("negative_prompt") or "").strip() \
            if float(ctx.get("cfg", 1.0) or 1.0) > 1.05 else ""
        negative = engine.call1("Krea2EditGroundedEncode", clip=clip, image=prepared,
                                prompt=_neg_txt, grounding_px=int(ctx.grounding_px))
        patch_kw = dict(model=model, source_latent=source_latent, vae=vae,
                        source_image=prepared, target_latent=target,
                        ref_boost=float(ctx.ref_boost), ref_boost_a=1.0, fit_mode="fit")
        if ref is not None:
            r = engine.call1("ImageScaleToTotalPixels", image=ref, upscale_method="lanczos",
                             megapixels=1.4, resolution_steps=64)
            patch_kw["source_image_b"] = r
            patch_kw["source_latent_b"] = engine.call1("VAEEncode", pixels=r, vae=vae)
        sampling = engine.call1("Krea2EditModelPatch",
                                **engine.only_supported("Krea2EditModelPatch", patch_kw))

    sampled = engine.call1("KSampler", model=sampling, seed=int(ctx.seed),
                           steps=int(ctx.steps), cfg=float(ctx.cfg),
                           sampler_name=ctx.get("sampler") or "euler",
                           scheduler=ctx.get("scheduler") or "simple",
                           positive=positive, negative=negative,
                           latent_image=target, denoise=float(ctx.get("denoise", 1.0)))
    generated = engine.call1("VAEDecode", samples=sampled, vae=vae)

    # Maskless is the normal case here (clothing changes measured at 11.3% of frame).
    # An all-zero mask lets 'smart' fall through to local-auto / full-frame.
    mask = ctx.get("mask")
    if mask is None:
        mask = _empty_mask_like(source)

    return engine.call1(
        "KreaImageUniversalAspectPreserveRestore",
        original_image=source, generated_image=generated, edit_mask=mask,
        restore_map=restore_map,
        restore_mode=ctx.get("restore_mode") or "smart: manual mask, local auto, or full frame",
        difference_threshold=0.05, difference_softness=0.02, max_local_coverage=0.45,
        mask_grow=12, mask_feather=10, local_color_match_strength=0.2,
    ), source


def _empty_mask_like(image):
    import torch
    b, h, w = image.shape[0], image.shape[1], image.shape[2]
    return torch.zeros((b, h, w), dtype=torch.float32, device="cpu")


# ---------------------------------------------------------------------------
# PIPELINE 3 — GREEN-MASK INPAINT / OUTPAINT
# ---------------------------------------------------------------------------

def green_mask_fill(ctx):
    source = need_image(ctx, "image")
    model, clip, vae = krea_base(ctx)
    model, _ = models.apply_loras(model, None, ctx.get("loras"))
    finalize_prompt(ctx, clip)

    if ctx.get("outpaint"):
        # OUTPAINT — no painting. The pad node produces both the canvas and the mask,
        # and the green border is composited from a solid green image.
        # Feathering softens the pad mask. Anything above ~8 leaves partially-green
        # pixels in the transition band, and the restore blends those back in as a
        # visible green seam — verified at feathering 24.
        canvas, mask = engine.call(
            "ImagePadForOutpaint", image=source,
            left=int(ctx.get("pad_left") or 0), top=int(ctx.get("pad_top") or 0),
            right=int(ctx.get("pad_right") or 0),
            bottom=int(ctx.get("pad_bottom") if ctx.get("pad_bottom") is not None else 256),
            feathering=int(ctx.get("pad_feather") if ctx.get("pad_feather") is not None else 8))[:2]
        w, h = engine.call("GetImageSize", image=canvas)[:2]
        green = engine.call1("EmptyImage", width=w, height=h, batch_size=1, color=GREEN)
        canvas = engine.call1("ImageCompositeMasked", destination=canvas, source=green,
                              x=0, y=0, resize_source=False, mask=mask)
    else:
        # INPAINT — the mask is the ALPHA channel you painted in MaskEditor, arriving
        # on the node's mask socket. It is grown, feathered, and handed to the prepare
        # node, which paints the green region itself. Green is never painted by hand.
        raw = ctx.get("mask")
        if raw is None:
            raise PipelineInputError(
                "INPAINT needs a mask. Connect the MASK output of your LoadImage node to "
                "the node's `mask` socket, and paint it with ComfyUI's MaskEditor "
                "(right-click the LoadImage node -> Open in MaskEditor). The mask is the "
                "image's alpha channel — do not paint green pixels yourself."
            )
        grown = engine.call1("GrowMask", mask=raw, expand=8, tapered_corners=True)
        as_img = engine.call1("MaskToImage", mask=grown)
        blurred = engine.call1("ImageBlur", image=as_img, blur_radius=6, sigma=4.0)
        mask = engine.call1("ImageToMask", image=blurred, channel="red")
        canvas = source

    # 7 outputs; restore_map is index 2, width 3, height 4 for THIS prepare node
    prep = engine.call("KreaAspectPreservePrepare", green_outpaint_canvas=canvas,
                       outpaint_mask=mask)
    prepared, restore_map, width, height = prep[0], prep[2], prep[3], prep[4]

    target = engine.call1("EmptyHunyuanLatentVideo", width=width, height=height,
                          length=1, batch_size=1)

    positive = engine.call1("TextEncodeKrea2OstrisEdit", clip=clip, vae=vae,
                            image1=prepared, prompt=ctx.prompt)
    _neg_txt = (ctx.get("negative_prompt") or "").strip() \
        if float(ctx.get("cfg", 1.0) or 1.0) > 1.05 else ""
    negative = engine.call1("TextEncodeKrea2OstrisEdit", clip=clip, vae=vae,
                            image1=prepared, prompt=_neg_txt)
    # REQUIRED — without this the reference latents are silently ignored.
    positive = engine.call1("FluxKontextMultiReferenceLatentMethod", conditioning=positive,
                            reference_latents_method="index_timestep_zero")
    negative = engine.call1("FluxKontextMultiReferenceLatentMethod", conditioning=negative,
                            reference_latents_method="index_timestep_zero")

    sampled = engine.call1("KSampler", model=model, seed=int(ctx.seed), steps=int(ctx.steps),
                           cfg=float(ctx.cfg), sampler_name=ctx.get("sampler") or "euler",
                           scheduler=ctx.get("scheduler") or "simple",
                           positive=positive, negative=negative,
                           latent_image=target, denoise=float(ctx.get("denoise", 1.0)))
    generated = engine.call1("VAEDecode", samples=sampled, vae=vae)

    final = engine.call1("KreaAspectPreserveRestore", expanded_original=canvas,
                         generated_image=generated, outpaint_mask=mask,
                         restore_map=restore_map, seam_overlap=6, seam_feather=8,
                         color_match_strength=0.25)
    return final, canvas


# ---------------------------------------------------------------------------
# PIPELINE 5 — FLUX 2 KLEIN DETAIL UPSCALE (a layer over 1-4)
# ---------------------------------------------------------------------------

def klein_upscale(ctx, image):
    model = models.unet(ctx.get("flux_unet_name") or FLUX_UNET)
    clip = models.clip(ctx.get("flux_clip_name") or FLUX_CLIP, FLUX_CLIP_TYPE)
    vae = models.vae(ctx.get("flux_vae_name") or FLUX_VAE)

    # ABSOLUTE megapixel target, not a multiplier
    scaled = engine.call1("ImageScaleToTotalPixels", image=image, upscale_method="lanczos",
                          megapixels=float(ctx.get("upscale_megapixels") or 2.0),
                          resolution_steps=16)
    w, h = engine.call("GetImageSize", image=scaled)[:2]
    latent = engine.call1("VAEEncode", pixels=scaled, vae=vae)

    cond = engine.call1("CLIPTextEncode", clip=clip,
                        text=ctx.get("upscale_prompt")
                        or "Upscale the image in high definition, restore realistic skin "
                           "texture, remove plastic looking skin, keep the entire image "
                           "content unchanged.")
    positive = engine.call1("ReferenceLatent", conditioning=cond, latent=latent)
    zeroed = engine.call1("ConditioningZeroOut", conditioning=cond)
    negative = engine.call1("ReferenceLatent", conditioning=zeroed, latent=latent)

    guider = engine.call1("CFGGuider", model=model, positive=positive, negative=negative,
                          cfg=float(ctx.get("upscale_cfg") or 1.0))
    sigmas = engine.call1("Flux2Scheduler", steps=int(ctx.get("upscale_steps") or 2),
                          width=w, height=h)
    sampler = engine.call1("KSamplerSelect", sampler_name="euler")
    noise = engine.call1("RandomNoise", noise_seed=int(ctx.seed))
    empty = engine.call1("EmptyFlux2LatentImage", width=w, height=h, batch_size=1)

    out = engine.call("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
                      sigmas=sigmas, latent_image=empty)[0]
    decoded = engine.call1("VAEDecode", samples=out, vae=vae)
    # ColorMatch (ComfyUI-KJNodes) and ImageSharpen are optional final polish — if a pack
    # isn't installed, skip that step rather than failing the whole upscale.
    matched = decoded
    if engine.has("ColorMatch"):
        matched = engine.call1("ColorMatch", image_ref=scaled, image_target=decoded,
                               method="mkl", strength=0.8, multithread=True)
    else:
        log.warning("[KreaAIO] ColorMatch (ComfyUI-KJNodes) not installed — skipping colour match.")
    if engine.has("ImageSharpen"):
        matched = engine.call1("ImageSharpen", image=matched, sharpen_radius=1, sigma=0.3, alpha=0.3)
    return matched
