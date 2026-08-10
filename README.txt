Krea2 AIO AJ  —  install notes
==============================

WHAT THIS IS
------------
One ComfyUI node that runs a whole 5-pipeline Krea 2 workflow internally:

  1  CLASSIC EDIT            two reference images
  2  IDENTITY / OSTRIS EDIT  MODE A (native Krea2Edit) or MODE B (ai-toolkit)
  3  GREEN-MASK              inpaint (you paint) or outpaint (auto green border)
  4  TEXT TO IMAGE
  5  FLUX 2 KLEIN UPSCALE    a layer on top of 1-4, not a separate pipeline

It does not reimplement anything. Every step calls the real ComfyUI node class
through ComfyUI's own registry, so the packs listed below must be installed.


INSTALL
-------
1. Copy the folder  KreaUltraController  into:

       ComfyUI/custom_nodes/

2. Restart ComfyUI completely (not just refresh the browser).

3. Open  example_workflows/Krea2_AIO_AJ.json  (a copy is placed in
   ComfyUI/user/default/workflows/ on install), or just drag the .json onto
   the ComfyUI canvas.

There is nothing to build, install or pip. It is plain Python + JavaScript.


TWO NODES IN THIS PACK
----------------------
  Krea2 AIO AJ          the 5-pipeline workflow node (below).
  Krea2 Live Preview AJ a large, resizable window that shows the sampler preview
                        WHILE generating, so you can watch the image form. Drop it
                        next to the AIO node. It needs live previews turned on:
                        Settings -> Preview method = Auto (or --preview-method auto).

The pack also bundles a GREEN MASK ENFORCER (js/krea_green_mask.js) that pins the
ComfyUI Mask Editor paint colour to neon green (#00FF00) reliably on any frontend
version - retrying past the open-time race, forcing adoption past value de-dup, and
guarding against resets. Because it ships with the pack, users do NOT need a separate
green-mask-colour node.


PER-WORKFLOW MEMORY (new)
-------------------------
Each of the 4 pipelines keeps its OWN prompt, sampler settings, denoise, seed +
seed mode (fixed/randomize), and LoRA stack. Switching tabs never shares a prompt
or resets another tab's settings. A fresh install starts with empty prompts. The
contextual guide at the top of the node changes to match the selected workflow and
includes brief prompting tips for editing, inpaint and outpaint.

LoRA stacks are per slot. Pipeline 2 keeps SEPARATE stacks for MODE A and MODE B,
so the krea2 identity-edit LoRA is kept ON by default in Classic (1) and Identity
MODE A (2A) - where it belongs - and is NOT carried into MODE B (ai-toolkit only).


DEPENDENCIES  (the core ones are now BUNDLED)
---------------------------------------------
The four core packs the pipelines rely on are shipped INSIDE this node (see
vendored/), so a fresh install of Krea2 AIO AJ runs classic edit, identity edit
(MODE A + B), inpaint, outpaint and text-to-image with NOTHING else to install:

  comfyui-krea2edit                 (Apache-2.0)  bundled
  ComfyUI-Krea2-Ostris-Edit         (MIT)         bundled
  ComfyUI-KreaImageAspectPreserve   (Apache-2.0)  bundled
  ComfyUI-KreaAspectPreserveOutpaint(Apache-2.0)  bundled

If you already have any of those installed separately, your copy is used instead
(the bundled copy only fills in what's missing — no conflicts). Credits + licenses
are in vendored/NOTICE.md and vendored/LICENSES/.

OPTIONAL packs — install only for the extra feature; the node runs fine without
them and just skips that step:

  ComfyUI-Impact-Pack + ComfyUI-Impact-Subpack   face detail pass
  ComfyUI-Easy-Use                               pipeline 1 background removal
  ComfyUI-KJNodes                                colour match on the Flux upscale
  rgthree-comfy                                  before/after comparer (example wf)

FluxKontextMultiReferenceLatentMethod, ImageSharpen, Flux2Scheduler, ResolutionSelector
and SamplerCustomAdvanced are ComfyUI core — nothing to install.

The optional links are inside the node itself, under "Optional extras & LoRAs".


LORAS  (direct downloads, into ComfyUI/models/loras/Krea2/)
-----------------------------------------------------------
  INPAINTKREA-V1            P3 inpaint/outpaint, 229 MB
    https://huggingface.co/Aitrepreneur/INPAINTKREA/resolve/main/INPAINTKREA-V1.safetensors
  krea2_identity_edit_v1_2  P2 MODE A, required for that mode
    https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/krea2_identity_edit_v1_2.safetensors


REQUIRED MODELS
---------------
Pipelines 1-4:
  diffusion_models/  Krea2/pornmasterKrea2_v2TurboInt8.safetensors
  text_encoders/     Qwen/qwen3vl_4b_fp8_scaled.safetensors     (CLIP type: krea2)
  vae/               wan/wan_2.1_vae.safetensors

Pipeline 5 (upscale only):
  diffusion_models/  Flux 2 9B/flux-2-klein-9b-fp8.safetensors
  text_encoders/     Qwen/qwen_3_8b_fp8mixed.safetensors         (CLIP type: flux2)
  vae/               Flux 2 9b/full_encoder_small_decoder.safetensors

LoRAs: see the LORAS section above. They are NOT interchangeable between
MODE A and MODE B.

If your paths differ, just pick your own files in the node's Models panel at the
top — nothing is hard-coded at runtime.


THINGS THAT FAIL SILENTLY  (read this)
--------------------------------------
* Krea 2 has NO default reference method. TextEncodeKrea2OstrisEdit on its own
  has its reference latents IGNORED — measured 0.11 correlation to the source vs
  0.997 with a method applied. Nothing errors. The node always applies exactly
  one method per mode, and never both.

* The LoRA must match the encoder. A MODE A LoRA under MODE B (or the reverse)
  degrades quietly. The node flags mismatches amber.

* The inpaint mask is the source image's ALPHA channel, not green pixels.
  ComfyUI's MaskEditor only ever saves an alpha mask. You never paint green
  yourself — KreaAspectPreservePrepare paints it from your mask.

* megapixels is an ABSOLUTE total-pixel target, not a multiplier.


USAGE
-----
Wire LoadImage -> image (and its MASK -> mask for inpaint). Pick a pipeline tab.
The node shows only the controls that pipeline actually uses. Its outputs are:

  image   the result      -> your SaveImage
  source  the original    -> Image Comparer input A (result goes to input B)

The node previews the result on itself but never saves; saving is your
SaveImage node's job, so you choose the folder and prefix.
