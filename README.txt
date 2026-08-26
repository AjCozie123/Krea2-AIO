Krea2 AIO  —  install notes
===========================

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
Easiest — git clone (recommended, so you can update with one command later).
Open a terminal / command prompt and run:

    cd ComfyUI/custom_nodes
    git clone https://github.com/AjCozie123/Krea2-AIO.git

  (On the ComfyUI portable build, that folder is usually
   ComfyUI_windows_portable/ComfyUI/custom_nodes.)
  This creates ComfyUI/custom_nodes/Krea2-AIO. Then FULLY restart ComfyUI —
  close it completely and reopen it, not just a browser refresh.

No git? On the GitHub page click the green "Code" button -> Download ZIP, unzip
  it, and drop the folder into ComfyUI/custom_nodes/. Restart ComfyUI.

TO UPDATE later to the newest version, run:

    cd ComfyUI/custom_nodes/Krea2-AIO
    git pull

  then FULLY restart ComfyUI.

Then open  example_workflows/Krea2_AIO.json  — or just drag that .json file onto
the ComfyUI canvas.

There is a SECOND example workflow next to it:

  example_workflows/Krea2_AIO_Expanded_Workflow.json

That is the exact same 5-pipeline setup rebuilt as an ORDINARY graph — every step
the AIO runs internally laid out as real nodes and noodles (164 of them), so you can
see, learn from, or modify any part of it. You switch pipeline there with the Fast
Groups Muter (rgthree) panel instead of the tab bar. It is for reading and tinkering;
the AIO node is still the quick way to actually work. It needs rgthree-comfy for the
group muter, the Any Switch nodes and the image comparer.

There is nothing to build, pip-install or compile — it is plain Python + JavaScript.


TWO NODES IN THIS PACK
----------------------
  Krea2 AIO             the 5-pipeline workflow node (below).
  Krea2 Live Preview    a large, resizable window that shows the sampler preview
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


NVIDIA RTX VSR  (optional final upscale)
----------------------------------------
In the node's MODELS panel there is an "OPTIONAL - NVIDIA RTX VSR" block with an
Enable tick. It is OFF by default and it is a CHOICE, not a requirement.

When ticked, the finished image gets one last pass through NVIDIA's RTX Video Super
Resolution after the VAE decode - and after the Flux 2 Klein upscale, if that layer
is also on. Two dials appear with the tick:

  Scale     multiplier, 1.0-4.0 (2.0 doubles each edge)
  Quality   LOW / MEDIUM / HIGH / ULTRA  (ULTRA is best and slowest)

It needs an NVIDIA RTX GPU, the nvidia-vfx runtime and a pack that provides the
RTXVideoSuperResolution node. If any of that is missing, or the VSR call fails, the
run logs a warning and hands back the image UN-upscaled - a finished generation is
never thrown away because an optional extra was unavailable. Leave the tick off and
nothing about the node changes.

Note: the two packs that provide this node register the SAME node id, so install
only one of them or ComfyUI will load whichever wins and ignore the other.


FREE VRAM / RAM  (optional tick, in the MODELS panel)
-----------------------------------------------------
Tick "OPTIONAL - FREE VRAM / RAM" and the node unloads models and hands freed memory
back at three points, chosen because nothing after them needs the model resident:

  1  after a SEPARATE enhancer LLM finishes  (the prompt is a plain string by then;
     skipped when the enhancer reuses the loaded encoder, since that would unload the
     very encoder about to be used)
  2  before the FLUX 2 KLEIN upscale  (a different model family is about to load, and
     it only consumes finished pixels - this is where 8 GB cards die)
  3  once the whole run is finished

It CANNOT change your image. It only unloads weights and returns freed blocks to the
driver; weights reload identically and nothing touches a latent, a seed or a
conditioning. The only cost is reload time.

It deliberately does NOT clear between the encoder and the sampler, between sampling
and the decode, or inside the face-detail pass - purging there unloads the very model
about to be used again, which is pure slowdown for no memory saved.

At the end of the run it also drops this node's OWN model cache (models.py keeps the
UNET / encoder / VAE in RAM between runs so it never re-reads from disk). That cache is
what leaves Krea 2 and Flux 2 sitting in system RAM after you are done with them, so
dropping it is what actually makes room for a heavy workflow like MiniMax H3 or LTX
afterwards. Leave the tick OFF if you are doing run after run in this node alone -
every purge means a reload.

The expanded example workflow has the same three points as FREE VRAM: groups holding
KJNodes' VRAM_Debug, with their own Fast Groups Muter panel to switch them on and off.


PROMPT ENHANCER (LLM, optional)
-------------------------------
Tick "Prompt Enhancer (LLM)" to rewrite your prompt before generating — the same
idea as ComfyUI's official Krea-2 workflow. It runs the core TextGenerate node on a
Qwen3-VL text encoder (by default the SAME one you already load, so no extra VRAM),
with a DIFFERENT system prompt per workflow: text-to-image expands into flowing Krea 2
prose; the edit workflows produce a short, grounded edit instruction. Pick the encoder
and a max-token size right in the node ("Text encoder - enhancer" + "Max tokens").

For uncensored expansion, choose an ABLITERATED Qwen3-VL in the encoder dropdown — the
stock qwen3vl model has its own refusals. If the enhancer errors, returns nothing, or
refuses, your typed prompt is used unchanged, so a run is never blocked.

Note: for the edit workflows the enhancer can be shown your reference image(s), but a
small 4B model is unreliable at image-grounded editing — text-to-image is where it
shines. Edit prompts are best kept short and written by hand anyway.

SEEING WHAT THE ENHANCER WROTE
There is a small round eye button next to the "Prompt Enhancer (LLM)" tick. Click it
after a run and it shows what you typed, what the LLM rewrote it into, and the final
prompt that was actually sampled with (trigger words are appended AFTER the rewrite,
which is why the last two can differ), plus a Copy button. The button picks up a blue
ring once there is something to show.

Because the enhancer falls back to your typed prompt on any error, empty result or
refusal, that panel also TELLS you when the rewrite did not happen — otherwise a
fallback is invisible and looks like the LLM simply chose not to change much.


DEPENDENCIES  (the core ones are now BUNDLED)
---------------------------------------------
The four core packs the pipelines rely on are shipped INSIDE this node (see
vendored/), so a fresh install of Krea2 AIO runs classic edit, identity edit
(MODE A + B), inpaint, outpaint and text-to-image with NOTHING else to install:

  comfyui-krea2edit                 (Apache-2.0)  bundled
  ComfyUI-Krea2-Ostris-Edit         (MIT)         bundled
  ComfyUI-KreaImageAspectPreserve   (Apache-2.0)  bundled
  ComfyUI-KreaAspectPreserveOutpaint(Apache-2.0)  bundled

If you already have any of those installed separately, your copy is used instead
(the bundled copy only fills in what's missing — no conflicts). Credits + licenses
are in vendored/NOTICE.md and vendored/LICENSES/.

OPTIONAL packs — install only for the extra feature; the node runs fine without
them and just skips that step. git clone into ComfyUI/custom_nodes/ (or install by
name in ComfyUI-Manager), then restart:

  face detail pass:
    https://github.com/ltdrdata/ComfyUI-Impact-Pack
    https://github.com/ltdrdata/ComfyUI-Impact-Subpack
    plus the detector model yolov12l-face.pt in models/ultralytics/bbox/ :
    https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov12l-face.pt
  pipeline 1 background removal:
    https://github.com/yolain/ComfyUI-Easy-Use
  colour match on the Flux upscale:
    https://github.com/kijai/ComfyUI-KJNodes
  before/after comparer (example workflow only):
    https://github.com/rgthree/rgthree-comfy
  NVIDIA RTX Video Super Resolution (final upscale, OFF by default):
    https://github.com/whmc76/ComfyUI-NVIDIA-RTX-VSR-Pro
    (or Comfy's stock Nvidia_RTX_Nodes_ComfyUI — either provides the node)
    also needs an NVIDIA RTX GPU and:  pip install nvidia-vfx

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

  image      the result      -> your SaveImage, and Image Comparer input B
  source     the BEFORE image -> Image Comparer input A. Normally the original input;
             when Face detail pass is ON it is the PRE-face-detail image, so the
             comparer shows exactly what the face pass changed.
  save_path  a name/prefix    -> wire into SaveImage.filename_prefix (optional)

The node previews the result on itself but never saves; saving is your
SaveImage node's job, so you choose the folder and prefix.
