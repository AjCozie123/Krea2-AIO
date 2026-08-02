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

3. Copy  "AJ AIO Node.json"  into:

       ComfyUI/user/default/workflows/

   or just drag the .json onto the ComfyUI canvas.

There is nothing to build, install or pip. It is plain Python + JavaScript.


REQUIRED CUSTOM NODE PACKS
--------------------------
Install these through ComfyUI-Manager. Without them the node still loads, but
the pipeline that needs a missing pack raises an error naming it.

  comfyui-krea2edit                  Krea2EditGroundedEncode, Krea2EditModelPatch
  ComfyUI-Krea2-Ostris-Edit          TextEncodeKrea2OstrisEdit
  ComfyUI-KreaImageAspectPreserve    KreaImageAspectPreservePrepare,
                                     KreaImageUniversalAspectPreserveRestore
  ComfyUI-KreaAspectPreserveOutpaint KreaAspectPreservePrepare,
                                     KreaAspectPreserveRestore
  rgthree-comfy                      Image Comparer
  comfyui-impact-pack + subpack      FaceDetailer, SAMLoader,
                                     UltralyticsDetectorProvider  (optional stage)
  comfyui-easy-use                   easy imageRemBg  (pipeline 1, off by default)

FluxKontextMultiReferenceLatentMethod, ColorMatch, ImageSharpen, Flux2Scheduler
and SamplerCustomAdvanced are ComfyUI core — nothing to install.


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

LoRAs (pick per mode, they are NOT interchangeable):
  MODE A  loras/Krea2/krea2_identity_edit_v1_2.safetensors
  MODE B  ai-toolkit LoRAs only, e.g. Krea2/INPAINTKREA-V1.safetensors

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
