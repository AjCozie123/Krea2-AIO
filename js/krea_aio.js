// KREA2 AJ AIO — embedded UI.
//
// The node runs all five pipelines internally (see pipelines.py), so this file does no
// graph manipulation at all. It hides the node's real widgets and renders a compact
// control surface that writes back into them, which is what reaches the backend.

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_W = 1300;  // horizontal rectangle — 3 columns so everything fits without scrolling
const NODE_H = 920;   // initial height before the DOM UI measures itself
// The node hugs its own content exactly (this is just a small floor). Node height = box + 64.
const NODE_MIN_BOX = 320;

const PIPES = [
  { idx: 1, n: "CLASSIC", sub: "2 refs", label: "1 - CLASSIC EDIT (two references)" },
  { idx: 2, n: "IDENTITY", sub: "edit", label: "2 - IDENTITY / OSTRIS EDIT" },
  { idx: 3, n: "IN/OUTPAINT", sub: "green mask", label: "3 - GREEN-MASK INPAINT / OUTPAINT" },
  { idx: 4, n: "TEXT2IMG", sub: "t2i", label: "4 - TEXT TO IMAGE" },
];

// Which controls actually drive something per pipeline. Hiding the rest keeps dials
// that do nothing off the screen.
const SHOW = {
  // Pipeline 1 samples into a ResolutionSelector-sized latent, so aspect_ratio and
  // megapixels are its resolution controls — they must be visible here.
  1: ["conn", "prompt", "seed", "steps", "cfg", "denoise", "sampler", "scheduler",
      "aspect_ratio", "megapixels", "size_mode", "grounding_px", "ref_boost", "loras",
      "face_detail", "remove_background", "use_reference"],
  2: ["conn", "prompt", "seed", "steps", "cfg", "denoise", "sampler", "scheduler",
      "grounding_px", "ref_boost", "restore_mode", "loras", "edit_mode",
      "use_reference", "aspect_ratio", "megapixels", "size_mode"],
  3: ["conn", "prompt", "seed", "steps", "cfg", "denoise", "sampler", "scheduler", "loras", "fill_mode"],
  4: ["prompt", "seed", "steps", "cfg", "denoise", "sampler", "scheduler", "aspect_ratio", "megapixels",
      "loras", "face_detail"],
};
const UPSCALE_EXTRA = ["megapixels"];

// Which sockets each pipeline actually consumes. `mask` is INPAINT-only by design:
// outpaint generates its own mask, and pipelines 1/2/4 ignore it entirely.
const SOCKETS = {
  1: [["image", true], ["reference", false]],
  2: [["image", true], ["mask", false], ["reference", false]],
  3: [["image", true], ["mask", "inpaint"]],
  4: [],
};

const PRESETS = {
  1: { steps: 8, cfg: 1.0, megapixels: 2.0, grounding_px: 768, ref_boost: 4.0 },
  2: { steps: 8, cfg: 1.0, megapixels: 1.0, grounding_px: 768, ref_boost: 1.0 },
  3: { steps: 8, cfg: 1.0, megapixels: 1.0, grounding_px: 768, ref_boost: 1.0 },
  4: { steps: 8, cfg: 1.0, megapixels: 2.0, grounding_px: 768, ref_boost: 1.0 },
};

// Every field that belongs to ONE pipeline. Each pipeline keeps its own copy in
// state_json, so switching tabs never shares a prompt or clobbers another tab's
// settings. `seed_control` maps to the seed's control_after_generate companion widget;
// everything else is a same-named widget. Models and the output folder are deliberately
// NOT here — they are global, shared across pipelines.
const PER_PIPE = [
  "prompt", "trigger_words", "seed", "seed_control", "steps", "cfg", "denoise", "sampler", "scheduler",
  "aspect_ratio", "megapixels", "size_mode", "grounding_px", "ref_boost", "restore_mode",
  "edit_mode", "fill_mode", "face_detail", "use_reference", "remove_background",
  "enhance_prompt", "negative_prompt",
];

// Per-workflow explanation of what the LLM prompt enhancer does (mirrors prompt_enhancer.py).
const ENH_INFO = {
  1: "Expands your prompt into a full description of the finished reference-edit image " +
     "(subject + scene), positive-only.",
  2: "Rewrites your request into a crisp Krea2Edit instruction — only what changes, keeping " +
     "everything else untouched.",
  3: "Rewrites your request into a description of just the masked / extended region so it " +
     "blends with the kept image.",
  4: "Expands your prompt into a grounded, parseable text-to-image description in Krea 2's " +
     "flowing-prose format.",
};
// NOTE: loras are NOT in PER_PIPE — they use their own per-slot store (see LORA_DEFAULTS
// and loraKey()), because pipeline 2 keeps a SEPARATE stack for MODE A vs MODE B.

const SEED_MODES = ["fixed", "increment", "decrement", "randomize"];

// Per-pipeline starting point on a FRESH node (empty prompts, sensible dials). PRESETS
// supplies the tuned numeric dials; the rest is a shared base.
const _PBASE = {
  prompt: "", seed: 0, seed_control: "randomize", steps: 8, cfg: 1.0, denoise: 1.0,
  sampler: "euler", scheduler: "simple",
  aspect_ratio: "3:4 (Portrait Standard)", megapixels: 1.0,
  grounding_px: 768, ref_boost: 1.0,
  restore_mode: "smart: manual mask, local auto, or full frame",
  edit_mode: "A - Native Krea2Edit (identity)",
  fill_mode: "A - INPAINT (you paint the mask)",
  face_detail: false, use_reference: true, remove_background: false,
  enhance_prompt: false, negative_prompt: "",
  // Default for pipelines 1 & 2: keep the edit at the input image's own size — what Krea2 edit
  // is tuned for. Switch to the aspect+megapixel option in the node to force a resolution.
  size_mode: "Match input image size (keeps the edit's own size)",
};
const PIPE_DEFAULTS = {
  1: { ..._PBASE, ...PRESETS[1] },
  2: { ..._PBASE, ...PRESETS[2] },
  3: { ..._PBASE, ...PRESETS[3] },
  4: { ..._PBASE, ...PRESETS[4] },
};

// LoRA stacks are stored per "slot", not simply per pipeline: pipeline 2 keeps SEPARATE
// stacks for MODE A and MODE B, because the identity LoRA belongs to MODE A only and
// silently degrades MODE B. Switching pipeline OR mode swaps the active stack.
// The krea2 identity-edit LoRA is kept ON by default in Classic (1) and Identity MODE A
// (2A); MODE B and the other pipelines start empty.
const IDENTITY_LORA = [{ on: true, lora: "Krea2\\krea2_identity_edit_v1_2.safetensors", strength: 1.0 }];
const LORA_DEFAULTS = {
  "1": JSON.stringify(IDENTITY_LORA),
  "2A": JSON.stringify(IDENTITY_LORA),
  "2B": "[]",
  "3": "[]",
  "4": "[]",
};

const MODE_A_LORA = "krea2_identity_edit_v1_2";
const MODE_B_LORAS = ["INPAINTKREA-V1", "Anime to Real", "zoom_krea2edit"];

// Crucial, measured notes. Several of these failure modes are silent.
const NOTES = {
  1: ["CLASSIC EDIT - two reference images", [
    ["How to prompt", "Describe the RESULT you want, not a command. Name the subject and the scene together, e.g. \"the woman from the reference wearing a red gown, standing in a sunlit garden\". Positive description only — say what should be there, not what to remove. Keep it to one or two sentences."],
    ["It regenerates the WHOLE frame", "Verified: this pipeline samples into a fresh ResolutionSelector-sized latent and has no aspect-preserve restore, so the background changes too. If you need the original background kept pixel-for-pixel, use pipeline 2 instead."],
    ["Reference order matters", "Vision blocks are fed in training order: image = scene/source, reference = subject/donor. Swapping them changes the result."],
    ["Reference method is REQUIRED", "Krea 2 has NO default reference method. This pipeline applies Krea2EditModelPatch. Without one the reference latents are silently IGNORED - measured 0.11 correlation to source vs 0.997 with. Nothing errors; the image just looks unrelated."],
    ["Never stack both methods", "Krea2EditModelPatch OR FluxKontextMultiReferenceLatentMethod. One, never both."],
    ["ref_boost", "Reference-fidelity dial. 1.0 = off; this pipeline is tuned at 4.0."],
    ["grounding_px", "Caps the longest side fed to Qwen3-VL. 512 here; 0 = native."],
    ["Remove background", "RMBG-2.0 on both inputs. Off by default - leave it off for scene-preserving edits like a clothing change."],
  ]],
  2: ["IDENTITY / OSTRIS EDIT - pick MODE, then match the LoRA", [
    ["How to prompt", "Write an EDIT INSTRUCTION describing only what changes, e.g. \"change the shirt to a black leather jacket\" or \"make the hair blonde\". Keep everything you do not mention untouched — do not re-describe the whole person. Short and specific beats long. For a clothing/body change use MODE A, maskless."],
    ["MODE A - Native Krea2Edit", "Best likeness, and the mode that preserves the untouched area exactly. Krea2EditGroundedEncode + Krea2EditModelPatch, then a native-resolution restore. LoRA must be krea2_identity_edit_v1_2."],
    ["MODE B - Ostris Edit", "TextEncodeKrea2OstrisEdit + FluxKontextMultiReferenceLatentMethod('index_timestep_zero'). That method is REQUIRED - the encoder alone has its reference latents ignored, silently. LoRAs: ai-toolkit only (INPAINTKREA-V1, Anime to Real, zoom_krea2edit)."],
    ["INPAINTKREA-V1 kills a maskless edit", "Measured: MODE B + INPAINTKREA-V1 on a maskless clothing instruction changed 0.0% of the frame; the same run with NO LoRA changed 51.9%. That LoRA only replaces green regions, so with no green present it correctly does nothing - and suppresses the edit. Use it in pipeline 3, not here."],
    ["LoRA must match the encoder", "A MODE A LoRA under MODE B (or the reverse) degrades quietly - no error, just a worse image. Mismatches are flagged amber in the LoRA list."],
    ["Clothing changes - use this", "Verified: MODE A, maskless, restore_mode 'smart'. Measured 11.3% of frame changed. Do NOT use a full-body mask in pipeline 3; those fail and leave green limbs."],
    ["restore_mode", "'smart' tries the manual mask, then local auto-detected edits, then the full frame. Everything outside the edit comes back at native resolution."],
    ["Resolution (optional here)", "Identity edit is tuned to the SOURCE image's own aspect and size, so it normally follows the source. You CAN set an Aspect + Megapixels here, but it works a bit LESS well — a different aspect can crop the framing and soften identity/edit adherence. For the best likeness keep it near the source's aspect; raise megapixels only if you want a larger result."],
  ]],
  3: ["GREEN-MASK INPAINT / OUTPAINT", [
    ["How to prompt - INPAINT", "Describe only what should APPEAR in the painted region, e.g. \"a leather jacket\" or \"a bunch of red roses\". Do not describe the rest of the image; only the mask is regenerated. Keep it short and concrete."],
    ["How to prompt - OUTPAINT", "Describe what should EXTEND beyond the current frame in the padded direction, e.g. \"more of the sandy beach and blue sky\". If you just want a natural continuation, a short scene description is enough."],
    ["How to paint the mask", "Right-click the SOURCE LoadImage node -> Open in MaskEditor, paint the region, save. Then wire that node's MASK output to this node's mask socket. That is the whole procedure."],
    ["You never paint green yourself", "The MASK is the image's ALPHA channel, not green pixels. ComfyUI's MaskEditor only ever saves an alpha mask - DefaultGreenMaskColor just recolours the swatch it shows you. KreaAspectPreservePrepare paints the actual green region from your alpha mask."],
    ["What the node does with it", "Your mask is grown by 8px, blurred (radius 6 / sigma 4) and fed back as a mask, so the green region has a soft edge. Then prepare paints it green, the Ostris encoder runs with index_timestep_zero, and KreaAspectPreserveRestore puts everything outside the mask back at native resolution."],
    ["OUTPAINT needs no mask", "It pads the canvas, generates its own mask from the padding, and composites the green border itself. The mask socket is ignored and hidden."],
    ["Do not mask whole bodies", "Full-body masks fail here and leave green limbs. For clothing or body changes use pipeline 2 MODE A maskless."],
    ["Removals", "Removals need a Raw checkpoint at cfg ~3 and ~20 steps - the turbo settings here will not remove cleanly."],
  ]],
  4: ["TEXT TO IMAGE", [
    ["How to prompt", "Full scene description: subject, setting, lighting and style, e.g. \"a golden retriever puppy on a wooden porch at sunset, warm soft light, photorealistic\". More detail = more control. No image needs to be wired."],
    ["Resolution", "Set by aspect ratio + megapixels, not by an input image. Keep near 1 MP on 8 GB VRAM."],
    ["megapixels is ABSOLUTE", "It is a total-pixel target, not a multiplier."],
    ["No image input", "This is the only pipeline that needs nothing wired to image / mask / reference."],
    ["Face detail", "Adds a FaceDetailer pass after the sampler."],
  ]],
  5: ["FLUX 2 KLEIN UPSCALE - a layer, not a pipeline", [
    ["It has no source of its own", "It refines whatever the selected pipeline produced, so a pipeline tab stays active underneath."],
    ["megapixels is ABSOLUTE", "4.0 means a 4 MP output regardless of input size. On 8 GB be careful above ~2 MP."],
    ["Separate models", "Uses its own Flux 2 Klein UNET / CLIP / VAE."],
  ]],
};

// Custom-node packs this node calls into. Links point to the repo PAGE so users can
// git clone them into ComfyUI/custom_nodes (LoRAs below are direct file downloads).
// [label, used by, repo url]  — null url = ships with ComfyUI core.
// The core edit/inpaint/outpaint/identity nodes are now BUNDLED inside this pack
// (see vendored/), so nothing below is required for the main pipelines. Only these
// OPTIONAL extras remain — install them just for the feature listed.
const DEPENDENCIES = [
  ["ComfyUI-Impact-Pack", "OPTIONAL · face detail", "https://github.com/ltdrdata/ComfyUI-Impact-Pack"],
  ["ComfyUI-Impact-Subpack", "OPTIONAL · face detail", "https://github.com/ltdrdata/ComfyUI-Impact-Subpack"],
  ["ComfyUI-Easy-Use", "OPTIONAL · P1 background removal", "https://github.com/yolain/ComfyUI-Easy-Use"],
  ["ComfyUI-KJNodes", "OPTIONAL · upscale colour match", "https://github.com/kijai/ComfyUI-KJNodes"],
  ["rgthree-comfy", "OPTIONAL · before/after comparer", "https://github.com/rgthree/rgthree-comfy"],
  ["ComfyUI-NVIDIA-RTX-VSR-Pro", "OPTIONAL · RTX VSR final upscale (needs an RTX GPU + pip install nvidia-vfx)",
   "https://github.com/whmc76/ComfyUI-NVIDIA-RTX-VSR-Pro"],
];

// Verified direct downloads; sizes checked against the local files.
const LORAS = [
  ["INPAINTKREA-V1", "P3 inpaint/outpaint",
   "https://huggingface.co/Aitrepreneur/INPAINTKREA/resolve/main/INPAINTKREA-V1.safetensors"],
  ["krea2_identity_edit_v1_2", "P2 MODE A",
   "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/krea2_identity_edit_v1_2.safetensors"],
];


// Mirrors build_save_path() in controller.py so the node can show where the result
// will land before you run it. The backend value is authoritative; this is a preview.
const SAVE_FOLDERS = { 1: ["Classic", "CE"], 2: ["Identity", "ID"],
                       3: ["Inpaint", "IN"], 4: ["TextToImage", "T2I"] };

function buildSavePath(root, idx, modeB, outpaint, faceDetail, upscale) {
  root = (root || "Krea2AJ").trim().replace(/^[\/]+|[\/]+$/g, "") || "Krea2AJ";
  let [folder, prefix] = SAVE_FOLDERS[idx] || ["Misc", "OUT"];
  if (idx === 2) [folder, prefix] = modeB ? ["Identity/OstrisB", "OSB"] : ["Identity/ModeA", "IDA"];
  else if (idx === 3) [folder, prefix] = outpaint ? ["Outpaint", "OUT"] : ["Inpaint", "IN"];
  if (faceDetail) prefix += "-fd";
  if (upscale) { folder += "/Upscaled"; prefix += "-4K"; }
  return `${root}/${folder}/${prefix}`;
}

const CSS = `
.kaio .foot{display:grid;grid-template-columns:1fr 1fr;gap:0 14px;margin-top:6px}
.kaio .refbtn{width:100%;text-align:left;padding:6px 9px;border-radius:7px;cursor:pointer;
 font-family:inherit;font-size:10.5px;background:#1d2530;border:1px solid #2c3d52;color:#9dc4ea;
 display:flex;align-items:center;gap:7px}
.kaio .refbtn:hover{background:#212b38}
.kaio .refbtn.plainbtn{background:var(--panel);border-color:var(--line);color:var(--dim);
 text-transform:uppercase;letter-spacing:.06em;font-size:9.5px}
.kaio .refbtn.plainbtn:hover{background:#2a2a2a;color:var(--txt)}
.kaio .refbtn i{font-style:italic;font-weight:700;display:inline-flex;align-items:center;
 justify-content:center;width:15px;height:15px;border-radius:50%;background:#2c3d52;flex:none}
.kaio .ovl{position:absolute;inset:0;background:rgba(22,22,22,.985);z-index:30;
 display:flex;flex-direction:column;border-radius:6px}
.kaio .ovl.hide{display:none!important}
.kaio .ovlhd{display:flex;align-items:center;gap:8px;padding:8px 10px;border-bottom:1px solid var(--line)}
.kaio .ovlhd b{flex:1;font-size:11px;color:#cfe3f7;font-weight:600}
.kaio .ovlhd button{padding:3px 10px;border-radius:5px;background:#2a2a2a;border:1px solid var(--line);
 color:var(--txt);cursor:pointer;font-size:11px;font-family:inherit}
.kaio .ovlhd button:hover{background:#3a3a3a}
.kaio .ovlbody{flex:1;overflow-y:auto;padding:6px 10px 10px}
.kaio{position:relative;
 --bg:#0d1017;--panel:#141a24;--panel2:#1a2130;--field:#0b0e14;--line:#26303f;
 --txt:#e7ecf3;--dim:#8791a1;--acc:#3b82f6;--acc2:#16233a;--acc-b:#4c8dff;
 --ok:#3b82f6;--ok2:#16233a;--warn:#e0a33e;
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:12px;color:var(--txt);
 background:linear-gradient(180deg,#0f141d 0%,#0b0e14 100%);border-radius:10px;
 padding:12px;box-sizing:border-box;width:100%;height:100%;overflow-y:auto;overflow-x:hidden;
}
.kaio > .kaio-inner{width:100%}
.kaio .cols{width:100%}
.kaio > .kaio-inner{display:block}
.kaio .cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0 12px;align-items:start}
.kaio .col{min-width:0}
/* Backstop for a panel that is forced narrower than we asked for (see applyWidth):
   fewer, readable columns instead of three squashed ones. */
.kaio.kaio-2col .cols{grid-template-columns:1fr 1fr}
.kaio.kaio-1col .cols{grid-template-columns:1fr}
.kaio .sec.grow{flex:0 0 auto;display:flex;flex-direction:column}
.kaio .sec.grow textarea{min-height:70px}
.kaio::-webkit-scrollbar{width:8px}.kaio::-webkit-scrollbar-thumb{background:#3a3a3a;border-radius:4px}
.kaio .hd{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.kaio .hd h1{font-size:12px;font-weight:600;letter-spacing:.08em;margin:0;flex:1;
 text-transform:uppercase;color:#fff}
.kaio .tabs{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-bottom:5px}
.kaio .tab{padding:6px 3px;border-radius:7px;background:var(--panel);border:1px solid var(--line);
 cursor:pointer;text-align:center;line-height:1.2;user-select:none;transition:.12s}
.kaio .tab:hover{background:#2a2a2a;border-color:#454545}
.kaio .tab.on{background:var(--acc2);border-color:var(--acc);color:#fff;box-shadow:0 0 0 1px var(--acc) inset}
.kaio .tab b{display:block;font-size:14px}.kaio .tab i{display:block;font-size:9.5px;opacity:.8;font-style:normal}
.kaio .upbar{width:100%;padding:5px;border-radius:7px;background:var(--panel);border:1px solid var(--line);
 color:var(--dim);cursor:pointer;font-size:11px;font-family:inherit;margin-bottom:6px;transition:.12s}
.kaio .upbar:hover{background:#2a2a2a}
.kaio .upbar.on{background:var(--acc2);border-color:var(--acc);color:#fff}
.kaio .sec{margin-bottom:6px}
.kaio label.cap{font-size:9px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);
 display:block;margin-bottom:2px}
.kaio .seg{display:flex;gap:5px}
.kaio .seg button{flex:1;padding:5px 6px;border-radius:6px;background:var(--panel);
 border:1px solid var(--line);color:var(--dim);cursor:pointer;font-size:10.5px;font-family:inherit;line-height:1.3}
.kaio .seg button:hover{background:#2a2a2a}
.kaio .seg button.on{background:var(--ok2);border-color:var(--ok);color:#fff}
.kaio .seg button b{display:block;font-size:11.5px}
.kaio .seg button i{font-style:normal;opacity:.75;font-size:9.5px}
.kaio input[type=text],.kaio input[type=number],.kaio select,.kaio textarea{
 background:var(--field);border:1px solid var(--line);border-radius:7px;color:var(--txt);
 padding:6px 8px;font-size:11.5px;font-family:inherit;box-sizing:border-box;width:100%}
.kaio input:focus,.kaio select:focus,.kaio textarea:focus{outline:none;border-color:var(--acc);
 box-shadow:0 0 0 2px rgba(59,130,246,.25)}
.kaio textarea{resize:vertical;min-height:52px;line-height:1.35}
.kaio .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}
.kaio .grid2{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.kaio .fld{display:flex;flex-direction:column;min-width:0}
.kaio .grouplab{font-size:9px;text-transform:uppercase;letter-spacing:.07em;color:#7fb0dd;
 display:block;margin:4px 0 4px;font-weight:700}
.kaio details.card.guide{margin-bottom:7px}
.kaio details.card.guide>summary{font-size:11px;color:#bcd8f2;font-weight:600}
.kaio .hide{display:none!important}
.kaio details.card{background:#1d2530;border:1px solid #2c3d52;border-radius:7px;margin-bottom:0;overflow:hidden}
.kaio details.card>summary{cursor:pointer;padding:7px 10px;font-size:11px;color:#9dc4ea;
 list-style:none;display:flex;align-items:center;gap:7px;user-select:none}
.kaio details.card>summary::-webkit-details-marker{display:none}
.kaio details.card>summary::before{content:"i";display:inline-flex;align-items:center;
 justify-content:center;width:15px;height:15px;border-radius:50%;background:#2c3d52;color:#9dc4ea;
 font-size:10px;font-style:italic;font-weight:700;flex:none}
.kaio details.card>summary:hover{background:#212b38}
.kaio details.card[open]>summary{border-bottom:1px solid #2c3d52}
.kaio details.card .body{padding:3px 10px 9px}
.kaio dl{margin:0}
.kaio dt{font-size:10.5px;font-weight:600;color:#bcd8f2;margin-top:7px}
.kaio dd{margin:2px 0 0;font-size:10.5px;line-height:1.5;color:#9aa8b6}
.kaio details.plain{background:var(--panel);border:1px solid var(--line);border-radius:7px;
 margin-bottom:0;overflow:hidden}
.kaio details.plain>summary{cursor:pointer;padding:7px 10px;font-size:10px;color:var(--dim);
 list-style:none;text-transform:uppercase;letter-spacing:.06em;user-select:none}
.kaio details.plain>summary::-webkit-details-marker{display:none}
.kaio details.plain>summary:hover{background:#282828}
.kaio details.models{background:#232823;border-color:#3a463a}
.kaio details.models>summary{color:#9fc39f;font-weight:600}
.kaio details.models>summary:hover{background:#283028}
.kaio details.plain .body{padding:4px 10px 9px}
.kaio .slot{display:flex;gap:8px;background:var(--panel);border:1px solid var(--line);
 border-radius:7px;padding:5px;margin-bottom:5px}
.kaio .thumb{width:62px;height:62px;border-radius:5px;background:#111;border:1px solid var(--line);
 object-fit:cover;flex:none}
.kaio .slotbody{flex:1;min-width:0;display:flex;flex-direction:column;gap:4px}
.kaio .slotttl{font-size:10px;color:var(--dim)}
.kaio .btnrow{display:flex;gap:5px}
.kaio .btnrow button{flex:1;padding:4px;border-radius:4px;background:#2a2a2a;border:1px solid var(--line);
 color:var(--dim);cursor:pointer;font-size:10px;font-family:inherit}
.kaio .btnrow button:hover{background:#333;color:var(--txt)}
.kaio .lrow{display:flex;align-items:center;gap:7px;background:var(--panel);border:1px solid var(--line);
 border-radius:5px;padding:4px 7px;margin-bottom:4px}
.kaio .lrow.off{opacity:.42}
.kaio .lrow input[type=checkbox]{width:auto;flex:none;margin:0;accent-color:var(--acc)}
.kaio .lname{flex:1 1 auto;min-width:0;font-size:10.5px;overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap}
.kaio .lname.bad{color:var(--warn)}
.kaio .lrow input.lstr{width:58px;flex:none;padding:3px 5px;font-size:10.5px}
.kaio .lx{flex:none;width:20px;height:20px;border-radius:4px;background:#2a2a2a;border:1px solid var(--line);
 color:var(--dim);cursor:pointer;font-size:12px;line-height:1;padding:0}
.kaio .lx:hover{background:#5a2a2a;color:#fff;border-color:#8a3a3a}
/* LoRA chain-order controls: position badge + move up/down */
.kaio .lord{flex:none;width:15px;text-align:center;font-size:9px;color:#6f8bb0;
 font-family:ui-monospace,Consolas,monospace}
.kaio .lmv{flex:none;width:17px;height:20px;border-radius:4px;background:#2a2a2a;
 border:1px solid var(--line);color:var(--dim);cursor:pointer;font-size:8px;line-height:1;padding:0}
.kaio .lmv:hover:not(:disabled){background:#2c3d52;color:#cfe3f7;border-color:var(--acc)}
.kaio .lmv:disabled{opacity:.25;cursor:default}
.kaio .addbtn{width:100%;padding:6px;border-radius:5px;background:#2a2a2a;border:1px dashed #444;
 color:var(--dim);cursor:pointer;font-size:11px;font-family:inherit}
.kaio .addbtn:hover{background:#303030;color:var(--txt);border-color:var(--acc)}
.kaio .picker{background:#141414;border:1px solid var(--acc);border-radius:6px;padding:6px;margin-top:5px}
.kaio .picker input{margin-bottom:5px}
.kaio .picklist{max-height:170px;overflow-y:auto}
.kaio .pickitem{padding:4px 6px;border-radius:4px;font-size:10.5px;cursor:pointer;
 overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kaio .pickitem:hover{background:var(--acc2);color:#fff}
.kaio .mrow{display:flex;align-items:center;gap:7px;margin-bottom:4px}
.kaio .mrow span{font-size:10px;color:var(--dim);width:52px;flex:none}
.kaio .mrow select{font-size:10.5px;padding:4px 5px}
/* Model-panel tick boxes (RTX VSR enable). Without this they inherit the generic
   full-width input rule and the browser's default grey, so a ticked box looks the
   same as an unticked one — they must read as ON/OFF like every other tick here. */
.kaio .mrow input[type=checkbox]{width:auto;flex:none;margin:0;accent-color:var(--acc)}
.kaio .chk{display:flex;align-items:center;gap:7px;font-size:11.5px;cursor:pointer}
.kaio .chk input{width:auto;accent-color:var(--acc)}
.kaio .status{font-size:10px;color:var(--dim);border-top:1px solid var(--line);padding-top:5px;
 margin-top:6px;line-height:1.45}
.kaio .status .warn{color:var(--warn);margin-top:3px}
.kaio .savepath{font-family:ui-monospace,Consolas,monospace;font-size:9.5px;color:#8fbf8f;
 margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kaio .savepath.unwired{color:var(--warn)}
.kaio .savepath.unwired::before{content:"! not wired - "}
.kaio .cmdrow{display:flex;gap:6px;align-items:center;margin:3px 0}
.kaio .sharenote{font-size:10px;color:var(--dim);line-height:1.5;margin-top:2px}
.kaio .deprow{display:flex;align-items:baseline;gap:8px;padding:2px 0}
.kaio .deplink{color:#7fb3e8;text-decoration:none;font-size:10.5px;flex:none}
.kaio .deplink:hover{text-decoration:underline;color:#a9cdf5}
.kaio .deplink.builtin{color:var(--dim);cursor:default}
.kaio .depuse{font-size:9.5px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kaio .muted{font-size:9.5px;color:var(--dim);margin:5px 0 3px}

/* ---------- polished "Krea 2 One Node" look ---------- */
.kaio .hd h1{font-size:13px;letter-spacing:.14em}
.kaio .tab{padding:8px 3px;border-radius:9px}
.kaio .tab.on{box-shadow:0 0 0 1px var(--acc) inset,0 0 12px rgba(59,130,246,.35)}
.kaio .seg button{padding:7px 6px;border-radius:8px}
.kaio .seg button.on{box-shadow:0 0 10px rgba(59,130,246,.30)}
.kaio .upbar{border-radius:9px;padding:8px}
.kaio .upbar.on{box-shadow:0 0 12px rgba(59,130,246,.35)}
.kaio .grouplab{color:#6f8bb0}
/* group the big sections into cards without touching build() — :has() is supported in
   the Chromium ComfyUI runs in */
.kaio .sec:has(> .grouplab),
.kaio .sec.grow,
.kaio .sec:has(.seg),
.kaio .sec:has(.addbtn),
.kaio .sec:has(.savepath){
 background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px;margin-bottom:7px}
.kaio details.plain,.kaio details.card{border-radius:11px}
/* range slider (LoRA strength) */
.kaio input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:5px;border-radius:4px;
 background:linear-gradient(90deg,var(--acc) 0%,var(--acc) var(--pct,100%),#243044 var(--pct,100%));
 outline:none;padding:0;border:none;margin:2px 0}
.kaio input[type=range]:focus{box-shadow:none}
.kaio input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;
 background:#eaf1ff;border:2px solid var(--acc);cursor:pointer;box-shadow:0 0 6px rgba(59,130,246,.5)}
/* LoRA cards (strength slider + trigger word) */
.kaio .lcard{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px 9px;margin-bottom:7px}
.kaio .lcard.off{opacity:.5}
.kaio .lcard-hd{display:flex;align-items:center;gap:8px}
.kaio .lcard-hd input[type=checkbox]{width:auto;flex:none;margin:0;accent-color:var(--acc)}
.kaio .lcard-hd .lname{flex:1 1 auto;min-width:0;font-size:11px;font-weight:600;overflow:hidden;
 text-overflow:ellipsis;white-space:nowrap}
.kaio .lcard-hd .lname.bad{color:var(--warn)}
.kaio .strrow{display:flex;align-items:center;gap:9px;margin-top:6px}
.kaio .strrow > label{font-size:9px;text-transform:uppercase;letter-spacing:.05em;color:var(--dim);flex:none;width:56px}
.kaio .strrow .sval{font-size:11px;color:var(--txt);width:36px;text-align:right;flex:none;
 font-family:ui-monospace,Consolas,monospace}
.kaio .trigrow{margin-top:7px}
.kaio .trigrow > label{font-size:9px;text-transform:uppercase;letter-spacing:.05em;color:var(--dim);
 display:block;margin-bottom:3px}
.kaio .trigrow input{font-size:10.5px}
.kaio .trigrow .thint{font-size:8.5px;color:#5f6b7c;margin-top:3px}
/* embedded live-preview box — fixed size, always present */
.kaio .kaio-preview{position:relative;width:100%;height:400px;background:#07090d;
 border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;
 align-items:center;justify-content:center}
.kaio .kaio-preview-img{max-width:100%;max-height:100%;object-fit:contain;display:none}
.kaio .kaio-preview-hint{position:absolute;color:#4a5468;font-size:10.5px;text-align:center;
 padding:12px;line-height:1.5;max-width:85%}
.kaio .kaio-preview-bar{position:absolute;left:0;bottom:0;height:3px;background:var(--acc);
 width:0%;transition:width .12s;box-shadow:0 0 8px rgba(59,130,246,.6)}
.kaio .kaio-preview-pct{position:absolute;right:6px;bottom:6px;font:10px ui-monospace,Consolas,monospace;
 color:#cfe0ff;background:rgba(0,0,0,.55);padding:1px 6px;border-radius:4px;display:none}
`;

function el(t, c, x) {
  const e = document.createElement(t);
  if (c) e.className = c;
  if (x != null) e.textContent = x;
  return e;
}

let LORA_LIST = null;
async function loraList() {
  if (LORA_LIST) return LORA_LIST;
  try {
    const r = await api.fetchApi("/object_info/LoraLoaderModelOnly");
    const d = await r.json();
    LORA_LIST = d.LoraLoaderModelOnly.input.required.lora_name[0] || [];
  } catch (e) {
    console.error("[KreaAIO] could not fetch lora list:", e);
    LORA_LIST = [];
  }
  return LORA_LIST;
}

const baseName = (p) => String(p).split(/[\\/]/).pop().replace(/\.safetensors$/i, "");

function thumbUrl(v) {
  let s = String(v || ""), type = "input";
  const m = s.match(/\s*\[(\w+)\]\s*$/);
  if (m) { type = m[1]; s = s.slice(0, m.index); }
  const i = s.lastIndexOf("/");
  const filename = i === -1 ? s : s.slice(i + 1);
  const subfolder = i === -1 ? "" : s.slice(0, i);
  if (!filename) return "";
  return api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=${type}` +
    `&subfolder=${encodeURIComponent(subfolder)}&rand=${Math.random()}`);
}

class AIO {
  constructor(node, root) {
    this.node = node;
    this.root = root;
    root.className = "kaio";
    this.applyWidth();   // before anything measures itself — see applyWidth()
    if (!document.getElementById("kaio-css")) {
      const s = el("style"); s.id = "kaio-css"; s.textContent = CSS;
      document.head.appendChild(s);
    }
    // All content lives in an auto-height inner wrapper. The root is the scroll box
    // (its height is driven by the node), so measuring the root tells us nothing —
    // the inner wrapper's height IS the content height.
    this.inner = el("div", "kaio-inner");
    root.appendChild(this.inner);

    this.ovl = el("div", "ovl hide");
    const oh = el("div", "ovlhd");
    this.ovlTitle = el("b");
    const close = el("button", null, "Close");
    close.onclick = () => { this.ovl.classList.add("hide"); this._guideOpen = false; };
    oh.appendChild(this.ovlTitle); oh.appendChild(close);
    this.ovlBody = el("div", "ovlbody");
    this.ovl.appendChild(oh); this.ovl.appendChild(this.ovlBody);
    root.appendChild(this.ovl);

    this.build();
    this.migrateOrInit();        // make sure every pipeline has its own saved slot
    this.loadLorasForCurrent();  // reflect the active slot's LoRA stack
    this.sync();

    // Re-fit when a <details> section is toggled, since that changes the height a lot.
    this.mBox.addEventListener("toggle", () => this.applyHeight());

    // Auto-fit the node to its content so NOTHING scrolls. The DOM widget element is the
    // node's whole body, so we grab it to keep its computeSize in step with the height we
    // pick. A ResizeObserver on the content wrapper re-fits whenever the visible controls
    // change (tab switch, LoRA added, guide toggled, …). Content height depends only on the
    // FIXED width, so reading it back to set the height can never become a growth loop.
    this._domWidget = this.node.widgets?.find((w) => w.name === "kaio_ui");
    this._fitBox = NODE_H - 64;
    try {
      this._ro = new ResizeObserver(() => this.applyHeight());
      this._ro.observe(this.inner);
    } catch (e) { /* no ResizeObserver: sync()/toggle still drive applyHeight() */ }
  }

  // Deterministic height from what is actually shown.
  //
  // Measuring the DOM does NOT work here: ComfyUI detaches/hides DOM widget elements
  // when the node is outside the viewport, so scrollHeight reads 0 and the node
  // collapses. Computing from the visible section list is independent of render state.
  estimateHeight() {
    const shown = (e) => e && !e.classList.contains("hide");
    let h = 8;
    h += 26;                                   // header
    h += 52 + 5;                               // pipeline tabs
    h += 33 + 8;                               // upscale bar
    const fluxRows = (this.fluxWrap && !this.fluxWrap.classList.contains("hide")) ? 5 * 30 + 18 : 0;
    // RTX VSR block: header + the always-visible tick row, plus 2 dial rows once ticked.
    const vsrRows = this.vsrWrap
      ? (30 + 18 + ((this.vsrCfg && !this.vsrCfg.classList.contains("hide")) ? 2 * 30 : 0))
      : 0;
    h += (this.mBox.open ? 30 + 3 * 30 + fluxRows + vsrRows : 32) + 9;   // models (top)
    h += 30 + 6;                               // notes button
    if (shown(this.modeSec)) h += 14 + 44 + 9;
    if (shown(this.fillSec)) h += 14 + 44 + 9;
    if (shown(this.padSec)) h += 14 + Math.ceil(5 / 3) * 52 + 9;
    if (shown(this.imgSec)) {
      h += 14 + this.connList.querySelectorAll(".lrow").length * 30;
      if (this.connList.querySelector(".muted")) h += 34;
      h += 9;
    }
    h += 14 + 78 + 9;                          // prompt
    h += 14 + 32 + 20 + 9;                     // output folder
    if (shown(this.refWrap)) h += 22;
    const nVis = Object.values(this.num).filter((x) => shown(x.f)).length;
    if (nVis) h += Math.ceil(nVis / 3) * 52 + 9;
    const cVis = Object.values(this.combo).filter((x) => shown(x.f)).length;
    if (cVis) h += Math.ceil(cVis / 3) * 52 + 9;
    if (shown(this.rSec)) h += 14 + 32 + 9;
    if (shown(this.lSec)) {
      h += 14 + Math.max(this.loras().length, 1) * 28 + 34 + 9;
      if (!this.picker.classList.contains("hide")) h += 210;
    }
    if (shown(this.faceSec)) h += 22 + 9;
    if (shown(this.rbSec)) h += 22 + 9;
    h += 0;                                    // links button shares the footer row
    h += 8 + 16 * (2 + this.status.querySelectorAll(".warn").length);
    return Math.min(Math.max(Math.round(h), 260), 1100);
  }

  // Own the panel's WIDTH instead of inheriting it.
  //
  // The whole layout rests on this one number: the 3-column grid and every height estimate
  // derive from it. We live inside a `.dom-widget` wrapper that ComfyUI sizes for us — and
  // for a node added FRESH from the node menu that wrapper is given `width:0; height:0` and
  // never revisited (measured on frontend 1.48.7). `.kaio{width:100%}` then resolves to
  // zero, `.cols` computes to "0px 0px 0px", and the panel is a dead 22px strip. It does not
  // self-heal on redraw. Loading a saved workflow happens to size the wrapper correctly,
  // which is the only reason this went unnoticed. Packs that patch the widget layer can
  // produce the same result.
  //
  // Overflowing the wrapper is safe: it computes `overflow:visible`, so nothing is clipped.
  applyWidth() {
    const root = this.root;
    if (!root) return;
    const want = NODE_W - 26;
    if (root.style.width !== want + "px") {
      root.style.width = want + "px";
      root.style.minWidth = want + "px";
    }
    // Backstop: if something still forces us narrower, drop to 2 or 1 columns so the panel
    // stays readable rather than squashing. A 0 reading means ComfyUI has detached the
    // element (it does that whenever the node is off-screen) — never collapse on a junk
    // measurement, just leave the current choice alone.
    let real = 0;
    try { real = parseFloat(getComputedStyle(root).width) || 0; } catch (e) { }
    if (real > 0) {
      root.classList.toggle("kaio-2col", real >= 620 && real < 980);
      root.classList.toggle("kaio-1col", real < 620);
    }
  }

  // Natural height is the floor, not the law: if the node has been dragged taller we
  // keep that and let the prompt box absorb the slack.
  // The node is a FIXED size and not resizable — see computeSize(). LiteGraph derives
  // node height from widget height, so anything that reads the node's own size back
  // into computeSize() compounds every frame and the node runs away down the canvas.
  // A constant cannot do that. Content scrolls inside instead.
  //
  // This only pins the size back to the constant; it never reads the current size into
  // the calculation, so it cannot become a feedback loop.
  applyHeight() {
    const n = this.node;
    if (!n) return;
    // Re-assert the width on every layout tick. This runs from the ResizeObserver, from
    // sync(), and from every <details> toggle, so anything that steals the width later gets
    // it taken back on the next tick. Must be BEFORE the early return below, which skips
    // out once the node size has settled.
    this.applyWidth();
    // Measure the real content height. ComfyUI detaches/hides DOM widget elements when the
    // node is off-screen, so scrollHeight can momentarily read ~0 — ignore those junk reads
    // and keep the last good value so the node never collapses.
    const measured = this.inner ? this.inner.scrollHeight : 0;
    if (measured > 120) {
      // +24 = the .kaio vertical padding (12 top + 12 bottom). The node is held at a LARGE
      // minimum height so it fills its area (and always has room for the Flux upscaler / enhancer
      // without scrolling); it only grows beyond that if a pipeline's content is taller still.
      this._fitBox = Math.min(Math.max(measured + 24, NODE_MIN_BOX), 2700);
    }
    const box = this._fitBox || (NODE_H - 64);
    const targetH = box + 64;  // + node title bar and input-socket rows
    // Width stays FIXED; only the height tracks content. Keep the DOM widget's own size in
    // step so the frontend gives the element the full box it needs.
    if (this._domWidget) this._domWidget.computeSize = () => [NODE_W - 26, box];
    if (n.size[0] === NODE_W && Math.abs(n.size[1] - targetH) <= 2) return;
    n.size[0] = NODE_W;
    n.size[1] = targetH;
    app.graph.setDirtyCanvas(true, true);
  }

  w(n) { return this.node.widgets?.find((x) => x.name === n); }
  setW(n, v) {
    const w = this.w(n);
    if (!w) return;
    w.value = v;
    if (w.callback) { try { w.callback(v); } catch (e) { } }
  }
  gv(n) { return this.w(n)?.value; }

  pipe() { return parseInt(String(this.gv("pipeline") || "4").trim()[0], 10) || 4; }
  upscale() { return !!this.gv("upscale"); }

  // ---- per-pipeline state ------------------------------------------------
  // state_json holds { "1":{…}, "2":{…}, "3":{…}, "4":{…} }. The real widgets always
  // mirror the ACTIVE pipeline (that is what the backend reads); this store is what
  // makes every other pipeline remember its own prompt, dials and LoRA stack.
  state() {
    try { const v = JSON.parse(this.gv("state_json") || "{}"); return (v && typeof v === "object") ? v : {}; }
    catch (e) { return {}; }
  }
  saveState(s) { this.setW("state_json", JSON.stringify(s)); }

  // Field name -> widget name. Only the seed control differs from its field name.
  wName(field) { return field === "seed_control" ? "control_after_generate" : field; }

  // Which LoRA slot is active. Pipeline 2 splits into MODE A / MODE B; the rest use the
  // pipeline number. This is what keeps the identity LoRA in 2A but not 2B.
  loraKey() {
    const p = this.pipe();
    if (p === 2) return String(this.gv("edit_mode") || "").startsWith("B") ? "2B" : "2A";
    return String(p);
  }

  // Load the current slot's LoRA stack into the live widget (on pipeline/mode switch).
  loadLorasForCurrent() {
    const s = this.state();
    const map = s._loras || {};
    const key = this.loraKey();
    const json = map[key] !== undefined ? map[key] : (LORA_DEFAULTS[key] ?? "[]");
    this.setW("loras_json", json);
  }

  // Write a per-pipeline field: update the live widget AND the active pipeline's slot.
  setP(field, value) {
    this.setW(this.wName(field), value);
    const s = this.state(), p = this.pipe();
    if (!s[p]) s[p] = { ...PIPE_DEFAULTS[p] };
    s[p][field] = value;
    this.saveState(s);
  }

  // Copy a pipeline's stored values into the live widgets (used when switching TO it).
  loadPipe(idx) {
    const s = this.state();
    const slot = s[idx] || PIPE_DEFAULTS[idx];
    for (const f of PER_PIPE) {
      if (slot[f] !== undefined) this.setW(this.wName(f), slot[f]);
    }
  }

  // Snapshot the live widgets back into the active pipeline's slot (used before we
  // switch away, and to capture a randomized seed after a run).
  captureActive() {
    const s = this.state(), p = this.pipe();
    if (!s[p]) s[p] = { ...PIPE_DEFAULTS[p] };
    for (const f of PER_PIPE) {
      const w = this.w(this.wName(f));
      if (w && w.value !== undefined) s[p][f] = w.value;
    }
    this.saveState(s);
  }

  // Ensure every pipeline has a slot. On a brand-new node the store is empty, so seed
  // the active pipeline from whatever the widgets currently hold (this migrates an
  // upgraded old workflow without wiping it) and give the rest their defaults.
  migrateOrInit() {
    let s = this.state();
    const fresh = !s || !Object.keys(s).length;
    const active = this.pipe();
    for (const idx of [1, 2, 3, 4]) {
      if (!s[idx]) {
        s[idx] = { ...PIPE_DEFAULTS[idx] };
        if (fresh && idx === active) {
          for (const f of PER_PIPE) {
            const w = this.w(this.wName(f));
            if (w && w.value !== undefined) s[idx][f] = w.value;
          }
        }
      }
    }
    // Per-slot LoRA stacks (with the identity LoRA kept in 1 and 2A by default).
    if (!s._loras) {
      s._loras = {};
      // Carry over an earlier version that stored a lora stack per pipeline slot.
      for (const idx of [1, 3, 4]) {
        if (s[idx] && s[idx].loras_json !== undefined) s._loras[String(idx)] = s[idx].loras_json;
      }
      if (s[2] && s[2].loras_json !== undefined) s._loras["2A"] = s[2].loras_json;
    }
    for (const k of Object.keys(LORA_DEFAULTS)) {
      if (s._loras[k] === undefined) s._loras[k] = LORA_DEFAULTS[k];
    }
    // Upgrade path: a pre-per-pipeline workflow with a single stack keeps it on the active slot.
    if (fresh) {
      const cur = this.gv("loras_json");
      if (cur && cur !== "[]") s._loras[this.loraKey()] = cur;
    }
    this.saveState(s);
  }

  build() {
    const r = this.inner;
    const hd = el("div", "hd");
    hd.appendChild(el("h1", null, "Krea2 All In One Workflow"));
    r.appendChild(hd);

    // Contextual guide, ABOVE the workflow row. An "i" button opens the help as a
    // rectangular bubble (overlay); clicking the same "i" again minimises it. Keeps the
    // panel clean — the help is out of the way until asked for, and follows the
    // selected workflow (contents set in sync()).
    this.guideBtn = el("button", "refbtn guidebtn");
    this.guideBtn.appendChild(el("i", null, "i"));
    this.guideTitle = el("span");
    this.guideBtn.appendChild(this.guideTitle);
    this.guideBody = el("div");
    this._guideOpen = false;
    this.guideBtn.onclick = () => {
      // toggle: if the guide bubble is showing, minimise it; otherwise open it
      if (this._guideOpen && !this.ovl.classList.contains("hide")) {
        this.ovl.classList.add("hide");
        this._guideOpen = false;
      } else {
        this.openOverlay(this.guideTitle.textContent, this.guideBody);
        this._guideOpen = true;
      }
    };
    r.appendChild(this.guideBtn);

    this.tabs = el("div", "tabs"); this.tabEl = {};
    for (const p of PIPES) {
      const t = el("div", "tab");
      t.appendChild(el("b", null, String(p.idx)));
      t.appendChild(el("i", null, p.n));
      t.title = p.label;
      // Save the pipeline we are leaving, switch, then load the target's own settings.
      // No preset() call here — switching must never reset a pipeline the user tuned.
      t.onclick = () => {
        this.captureActive();
        this.setW("pipeline", p.label);
        this.loadPipe(p.idx);
        this.loadLorasForCurrent();
        this.sync();
      };
      this.tabs.appendChild(t); this.tabEl[p.idx] = t;
    }
    r.appendChild(this.tabs);

    this.upBtn = el("button", "upbar", "5 · FLUX 2 KLEIN UPSCALE");
    this.upBtn.onclick = () => { this.setW("upscale", !this.upscale()); this.sync(); };
    r.appendChild(this.upBtn);

    // Two columns so the node reads as a square panel instead of a tall ribbon.
    const body = el("div", "cols");
    r.appendChild(body);
    const L = el("div", "col");
    const M = el("div", "col");
    const R = el("div", "col");
    body.appendChild(L); body.appendChild(M); body.appendChild(R);

    // Live preview — a FIXED-size box that is ALWAYS present, so a running generation
    // shows here without ever resizing the node. Sits top of the right column.
    this.prevSec = el("div", "sec previewsec");
    this.prevSec.appendChild(el("label", "grouplab", "Live preview"));
    const pv = el("div", "kaio-preview");
    this.prevImg = el("img", "kaio-preview-img");
    this.prevImg.onerror = () => { this.prevImg.style.display = "none"; };
    this.prevHint = el("div", "kaio-preview-hint",
      "Live sampler preview shows here while this workflow generates. " +
      "Blank? Enable Settings → Preview method → Auto (TAESD).");
    this.prevBar = el("div", "kaio-preview-bar");
    this.prevPct = el("div", "kaio-preview-pct");
    pv.appendChild(this.prevImg); pv.appendChild(this.prevHint);
    pv.appendChild(this.prevBar); pv.appendChild(this.prevPct);
    this.prevSec.appendChild(pv);
    R.appendChild(this.prevSec);
    this.setupLivePreview();

    // Models first — this is where you point it at your checkpoints, so it should be
    // the most obvious thing after picking a pipeline.
    this.mBox = el("details", "plain models");
    this.mBox.open = true;
    this.mBox.appendChild(el("summary", null, "Models"));
    this.mBody = el("div", "body"); this.mBox.appendChild(this.mBody);
    L.appendChild(this.mBox);
    this.modelRows = {};
    for (const [key, lbl, hint] of [
      ["unet_name", "Diffusion", "UNETLoader — the Krea 2 checkpoint"],
      ["clip_name", "Text enc.", "CLIPLoader — Qwen3-VL, type krea2"],
      ["vae_name", "VAE", "VAELoader"],
    ]) {
      const row = el("div", "mrow");
      const lab = el("span", null, lbl);
      lab.title = hint;
      row.appendChild(lab);
      const s = el("select");
      s.onchange = () => { this.setW(key, s.value); s.title = s.value; };
      row.appendChild(s);
      this.mBody.appendChild(row);
      this.modelRows[key] = s;
    }
    // Flux 2 Klein runs a different model family, so it gets its own loaders. Shown
    // only when the upscale is on, so they are not clutter the rest of the time.
    this.fluxWrap = el("div");
    this.fluxWrap.appendChild(el("div", "muted", "FLUX 2 KLEIN UPSCALE"));
    for (const [key, lbl] of [["flux_unet_name", "Diffusion"], ["flux_clip_name", "Text enc."],
                              ["flux_vae_name", "VAE"]]) {
      const row = el("div", "mrow");
      row.appendChild(el("span", null, lbl));
      const s = el("select");
      s.onchange = () => { this.setW(key, s.value); s.title = s.value; };
      row.appendChild(s);
      this.fluxWrap.appendChild(row);
      this.modelRows[key] = s;
    }
    const usRow = el("div", "mrow");
    usRow.appendChild(el("span", null, "Steps"));
    this.upSteps = el("input"); this.upSteps.type = "number";
    this.upSteps.min = 1; this.upSteps.max = 50;
    this.upSteps.oninput = () => this.setW("upscale_steps", Number(this.upSteps.value || 2));
    usRow.appendChild(this.upSteps);
    this.fluxWrap.appendChild(usRow);
    const umRow = el("div", "mrow");
    umRow.appendChild(el("span", null, "Megapix"));
    this.upMP = el("input"); this.upMP.type = "number";
    this.upMP.min = 0.5; this.upMP.max = 16; this.upMP.step = 0.5;
    this.upMP.title = "ABSOLUTE target. The reference workflow uses 4.0; lower it on 8 GB.";
    this.upMP.oninput = () => this.setW("upscale_megapixels", Number(this.upMP.value || 4));
    umRow.appendChild(this.upMP);
    this.fluxWrap.appendChild(umRow);
    this.mBody.appendChild(this.fluxWrap);

    // OPTIONAL · NVIDIA RTX Video Super Resolution. Runs last, after the decode and after
    // the Flux upscale. Deliberately a tick-box that is OFF by default: it needs an NVIDIA
    // RTX card, the nvidia-vfx runtime and the RTX VSR node pack, which many users won't
    // have. The tick row is always visible so the option is discoverable; its two dials
    // only appear once it's ticked, same idea as the Flux block above.
    this.vsrWrap = el("div");
    this.vsrWrap.appendChild(el("div", "muted", "OPTIONAL · NVIDIA RTX VSR"));
    const vsrOnRow = el("div", "mrow");
    vsrOnRow.appendChild(el("span", null, "Enable"));
    this.vsrChk = el("input"); this.vsrChk.type = "checkbox";
    this.vsrChk.title = "Upscale the finished image with NVIDIA RTX Video Super Resolution. " +
                        "Needs an RTX GPU, the nvidia-vfx package and the RTX VSR node pack. " +
                        "Leave off if you don't have those — the run is unaffected.";
    this.vsrChk.onchange = () => { this.setW("rtx_vsr", this.vsrChk.checked); this.sync(); };
    vsrOnRow.appendChild(this.vsrChk);
    this.vsrWrap.appendChild(vsrOnRow);

    this.vsrCfg = el("div");
    const vsRow = el("div", "mrow");
    vsRow.appendChild(el("span", null, "Scale"));
    this.vsrScale = el("input"); this.vsrScale.type = "number";
    this.vsrScale.min = 1; this.vsrScale.max = 4; this.vsrScale.step = 0.25;
    this.vsrScale.title = "Multiplier. 2.0 doubles each edge.";
    this.vsrScale.oninput = () => this.setW("rtx_vsr_scale", Number(this.vsrScale.value || 2));
    vsRow.appendChild(this.vsrScale);
    this.vsrCfg.appendChild(vsRow);
    const vqRow = el("div", "mrow");
    vqRow.appendChild(el("span", null, "Quality"));
    this.vsrQual = el("select");
    this.vsrQual.onchange = () => this.setW("rtx_vsr_quality", this.vsrQual.value);
    vqRow.appendChild(this.vsrQual);
    this.vsrCfg.appendChild(vqRow);
    this.vsrWrap.appendChild(this.vsrCfg);
    this.mBody.appendChild(this.vsrWrap);

    // The per-workflow guide now lives at the top of the panel (built above). The footer
    // holds only the download-links button, which opens as an OVERLAY — inside a
    // fixed-height node an inline expander would push controls out of view.
    this.foot = el("div", "foot");   // appended near the end so it sits at the bottom

    // MODE A/B
    this.modeSec = el("div", "sec");
    this.modeSec.appendChild(el("label", "cap", "Edit mode — encoder + LoRA family"));
    const ms = el("div", "seg");
    this.mA = el("button"); this.mA.appendChild(el("b", null, "MODE A"));
    this.mA.appendChild(el("i", null, "Native Krea2Edit"));
    this.mB = el("button"); this.mB.appendChild(el("b", null, "MODE B"));
    this.mB.appendChild(el("i", null, "Ostris · ai-toolkit"));
    // Switching mode in pipeline 2 swaps to that mode's own LoRA stack (2A keeps the
    // identity LoRA; 2B does not).
    this.mA.onclick = () => { this.setP("edit_mode", "A - Native Krea2Edit (identity)"); this.loadLorasForCurrent(); this.sync(); };
    this.mB.onclick = () => { this.setP("edit_mode", "B - Ostris Edit (ai-toolkit)"); this.loadLorasForCurrent(); this.sync(); };
    ms.appendChild(this.mA); ms.appendChild(this.mB);
    this.modeSec.appendChild(ms); L.appendChild(this.modeSec);

    // FILL A/B
    this.fillSec = el("div", "sec");
    this.fillSec.appendChild(el("label", "cap", "Inpaint or outpaint"));
    const fs = el("div", "seg");
    this.fA = el("button"); this.fA.appendChild(el("b", null, "INPAINT"));
    this.fA.appendChild(el("i", null, "you paint the mask"));
    this.fB = el("button"); this.fB.appendChild(el("b", null, "OUTPAINT"));
    this.fB.appendChild(el("i", null, "auto green border"));
    this.fA.onclick = () => { this.setP("fill_mode", "A - INPAINT (you paint the mask)"); this.sync(); };
    this.fB.onclick = () => { this.setP("fill_mode", "B - OUTPAINT (auto green border)"); this.sync(); };
    fs.appendChild(this.fA); fs.appendChild(this.fB);
    this.fillSec.appendChild(fs); L.appendChild(this.fillSec);

    // outpaint padding — only meaningful for pipeline 3 OUTPAINT
    this.padSec = el("div", "sec");
    this.padSec.appendChild(el("label", "cap", "Outpaint padding (px)"));
    const pg = el("div", "grid");
    this.pad = {};
    for (const [key, lbl] of [["outpaint_bottom", "Bottom"], ["outpaint_top", "Top"],
                              ["outpaint_left", "Left"], ["outpaint_right", "Right"],
                              ["outpaint_feather", "Feather"]]) {
      const f = el("div", "fld");
      f.appendChild(el("label", "cap", lbl));
      const i = el("input"); i.type = "number"; i.min = 0; i.step = key.includes("feather") ? 1 : 8;
      i.oninput = () => this.setW(key, Number(i.value || 0));
      if (key.includes("feather")) {
        i.title = "Above ~8 leaves partially-green pixels in the transition band, which " +
                  "show up as a green seam after the restore.";
      }
      f.appendChild(i);
      this.pad[key] = { f, i };
      pg.appendChild(f);
    }
    this.padSec.appendChild(pg);
    M.appendChild(this.padSec);

    // input sockets — images live on real LoadImage nodes outside this one, which is
    // what keeps ComfyUI's MaskEditor working normally
    this.imgSec = el("div", "sec");
    this.imgSec.appendChild(el("label", "cap", "Inputs — wire LoadImage nodes here"));
    this.connList = el("div");
    this.imgSec.appendChild(this.connList);
    this.refWrap = el("label", "chk");
    this.refWrap.style.marginTop = "5px";
    this.useRef = el("input"); this.useRef.type = "checkbox";
    this.useRef.onchange = () => { this.setP("use_reference", this.useRef.checked); this.sync(); };
    this.refWrap.appendChild(this.useRef);
    this.refWrap.appendChild(el("span", null, "Use second reference image"));
    this.imgSec.appendChild(this.refWrap);
    L.appendChild(this.imgSec);

    // prompt (per-pipeline — each workflow keeps its own instruction)
    this.pSec = el("div", "sec grow");
    this.pSec.appendChild(el("label", "cap", "Prompt / instruction"));
    this.prompt = el("textarea");
    this.prompt.oninput = () => this.setP("prompt", this.prompt.value);
    this.pSec.appendChild(this.prompt);
    // Negative prompt. Collapsed by default because it is INERT on Krea 2 Turbo (cfg 1.0) —
    // the summary says so, so nobody wastes an hour writing negatives that cannot apply.
    this.negBox = el("details", "plain");
    this.negSum = el("summary", null, "Negative prompt — needs CFG above 1.0");
    this.negBox.appendChild(this.negSum);
    const negBody = el("div", "body");
    this.negPrompt = el("textarea"); this.negPrompt.rows = 2;
    this.negPrompt.style.minHeight = "38px";
    this.negPrompt.placeholder = "e.g. shallow depth of field, bokeh, blurred foreground";
    this.negPrompt.oninput = () => this.setP("negative_prompt", this.negPrompt.value);
    negBody.appendChild(this.negPrompt);
    this.negHint = el("div", "muted", "");
    negBody.appendChild(this.negHint);
    this.negBox.appendChild(negBody);
    this.pSec.appendChild(this.negBox);
    r.insertBefore(this.pSec, body);  // full-width, on top

    // Shared field builders. Every value is written per-pipeline through setP, so it is
    // remembered for that workflow and never leaks into another tab.
    this.num = {}; this.combo = {};
    const mk = (grid, name, label, step, min, max) => {
      const f = el("div", "fld");
      f.appendChild(el("label", "cap", label));
      const i = el("input"); i.type = "number";
      if (step != null) i.step = step; if (min != null) i.min = min; if (max != null) i.max = max;
      i.oninput = () => this.setP(name, i.value === "" ? 0 : Number(i.value));
      f.appendChild(i); this.num[name] = { f, i }; grid.appendChild(f);
    };
    const mkc = (grid, name, label) => {
      const f = el("div", "fld");
      f.appendChild(el("label", "cap", label));
      const s = el("select");
      s.onchange = () => this.setP(name, s.value);
      f.appendChild(s); this.combo[name] = { f, s }; grid.appendChild(f);
    };

    // ---- SAMPLER settings, grouped and in the order that matters ----
    this.sampSec = el("div", "sec");
    this.sampSec.appendChild(el("label", "grouplab", "Sampler settings"));
    const seedRow = el("div", "grid2");
    mk(seedRow, "seed", "Seed", 1, 0);
    // Seed control: lock a result (fixed) or re-roll it (randomize) between runs.
    const smf = el("div", "fld");
    smf.appendChild(el("label", "cap", "After generate"));
    this.seedMode = el("select");
    this.seedMode.title = "fixed = keep this seed to retest a similar image; " +
                          "randomize = new seed each run.";
    for (const m of SEED_MODES) { const o = el("option", null, m); o.value = m; this.seedMode.appendChild(o); }
    this.seedMode.onchange = () => this.setP("seed_control", this.seedMode.value);
    smf.appendChild(this.seedMode); seedRow.appendChild(smf);
    this.sampSec.appendChild(seedRow);
    const scRow = el("div", "grid");
    mk(scRow, "steps", "Steps", 1, 1, 200);
    mk(scRow, "cfg", "CFG", 0.1, 0, 30);
    mk(scRow, "denoise", "Denoise", 0.01, 0, 1);
    this.sampSec.appendChild(scRow);
    const ssRow = el("div", "grid2");
    mkc(ssRow, "sampler", "Sampler");
    mkc(ssRow, "scheduler", "Scheduler");
    this.sampSec.appendChild(ssRow);
    M.appendChild(this.sampSec);

    // ---- RESOLUTION ----
    this.resSec = el("div", "sec");
    this.resSec.appendChild(el("label", "grouplab", "Resolution"));
    // Pipelines 1 & 2 choose HOW the size is decided: keep the input image's own size (best for
    // identity / edit adherence) or force an aspect + megapixel target. Hidden for 3 and 4.
    this.sizeSec = el("div", "fld");
    this.sizeSec.style.marginBottom = "5px";
    this.sizeSec.appendChild(el("label", "cap", "Output size"));
    this.sizeSel = el("select");
    this.sizeSel.title = "Match input image size = the edit comes out at the source image's own "
      + "dimensions (best identity / edit adherence). Aspect ratio + megapixels = force the "
      + "resolution you pick below.";
    this.sizeSel.onchange = () => { this.setP("size_mode", this.sizeSel.value); this.sync(); };
    this.sizeSec.appendChild(this.sizeSel);
    this.resSec.appendChild(this.sizeSec);
    const resRow = el("div", "grid2");
    mkc(resRow, "aspect_ratio", "Aspect");
    mk(resRow, "megapixels", "Megapixels", 0.1, 0.1, 16);
    this.resSec.appendChild(resRow);
    M.appendChild(this.resSec);

    // ---- REFERENCE dials (pipeline 1 and pipeline 2 MODE A) ----
    this.refSec = el("div", "sec");
    this.refSec.appendChild(el("label", "grouplab", "Reference"));
    const refRow = el("div", "grid2");
    mk(refRow, "grounding_px", "Grounding px", 64, 0, 4096);
    mk(refRow, "ref_boost", "Ref boost", 0.01, 0, 1000);
    this.refSec.appendChild(refRow);
    M.appendChild(this.refSec);

    // restore (per-pipeline)
    this.rSec = el("div", "sec");
    this.rSec.appendChild(el("label", "cap", "Restore mode"));
    this.restore = el("select");
    this.restore.onchange = () => this.setP("restore_mode", this.restore.value);
    this.rSec.appendChild(this.restore); M.appendChild(this.rSec);

    // loras
    this.lSec = el("div", "sec");
    this.lSec.appendChild(el("label", "cap", "LoRA stack"));
    this.lList = el("div");
    this.lSec.appendChild(this.lList);
    this.addBtn = el("button", "addbtn", "+  Add LoRA");
    this.addBtn.onclick = () => this.togglePicker();
    this.lSec.appendChild(this.addBtn);
    this.picker = el("div", "picker"); this.picker.classList.add("hide");
    this.pickInput = el("input"); this.pickInput.type = "text";
    this.pickInput.placeholder = "search LoRAs…";
    this.pickInput.oninput = () => this.renderPicker();
    this.pickList = el("div", "picklist");
    this.picker.appendChild(this.pickInput); this.picker.appendChild(this.pickList);
    this.lSec.appendChild(this.picker);
    // ONE trigger-words box for all LoRAs — added to the prompt automatically.
    const tw = el("div", "sec"); tw.style.marginTop = "6px";
    tw.appendChild(el("label", "cap", "LoRA trigger words (added to prompt)"));
    this.trigWords = el("textarea"); this.trigWords.rows = 2;
    this.trigWords.style.minHeight = "34px";
    this.trigWords.placeholder = "all your trigger words here…";
    this.trigWords.oninput = () => this.setP("trigger_words", this.trigWords.value);
    tw.appendChild(this.trigWords);
    this.lSec.appendChild(tw);
    L.appendChild(this.lSec);   // LoRAs on the left, below models & reference inputs

    // toggles
    this.faceSec = el("div", "sec");
    const fl = el("label", "chk");
    this.face = el("input"); this.face.type = "checkbox";
    this.face.onchange = () => { this.setP("face_detail", this.face.checked); this.sync(); };
    fl.appendChild(this.face); fl.appendChild(el("span", null, "Face detail pass"));
    this.faceSec.appendChild(fl);
    // Face detail needs a YOLO face model file — small clickable link to grab it.
    const fdnote = el("div", "muted");
    fdnote.appendChild(document.createTextNode("requires yolov12l-face.pt in models/ultralytics/bbox/ — "));
    const fda = el("a", "deplink", "download .pt");
    fda.href = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov12l-face.pt";
    fda.target = "_blank"; fda.rel = "noopener";
    fda.title = "yolov12l-face.pt — required model for the face detail pass";
    fdnote.appendChild(fda);
    this.faceSec.appendChild(fdnote);
    M.appendChild(this.faceSec);

    this.rbSec = el("div", "sec");
    const rbl = el("label", "chk");
    this.rb = el("input"); this.rb.type = "checkbox";
    this.rb.onchange = () => this.setP("remove_background", this.rb.checked);
    rbl.appendChild(this.rb);
    rbl.appendChild(el("span", null, "Remove background (RMBG-2.0)"));
    this.rbSec.appendChild(rbl); M.appendChild(this.rbSec);

    // ---- PROMPT ENHANCER (LLM) ----
    // The Krea2-style enhancer: rewrites the prompt with core TextGenerate on the loaded
    // Qwen3-VL, using a per-workflow system prompt (backend prompt_enhancer.py). A tick to
    // enable/disable, an "i" that explains what it does for THIS workflow, and (when on) the
    // model + token controls.
    this.enhSec = el("div", "sec");
    this.enhSec.appendChild(el("label", "grouplab", "Prompt enhancer"));
    const ehHd = el("label", "chk");
    this.enhChk = el("input"); this.enhChk.type = "checkbox";
    this.enhChk.onchange = () => { this.setP("enhance_prompt", this.enhChk.checked); this.sync(); };
    ehHd.appendChild(this.enhChk);
    ehHd.appendChild(el("span", null, "Prompt Enhancer (LLM)"));
    this.enhSec.appendChild(ehHd);
    this.enhInfoBtn = el("button", "refbtn");
    this.enhInfoBtn.style.marginTop = "5px";
    this.enhInfoBtn.appendChild(el("i", null, "i"));
    this.enhInfoBtn.appendChild(el("span", null, "What does this do?"));
    this.enhInfoBtn.onclick = () => this.openOverlay("Prompt enhancer (LLM)", this.buildEnhInfo());
    this.enhSec.appendChild(this.enhInfoBtn);
    // model + tokens — shown only when the enhancer is enabled
    this.enhCfg = el("div"); this.enhCfg.style.marginTop = "6px";
    const emf = el("div", "fld");
    emf.appendChild(el("label", "cap", "Text encoder - enhancer"));
    this.enhModel = el("select");
    this.enhModel.onchange = () => { this.setW("enhancer_model", this.enhModel.value); this.enhModel.title = this.enhModel.value; };
    emf.appendChild(this.enhModel); this.enhCfg.appendChild(emf);
    const etf = el("div", "fld"); etf.style.marginTop = "5px";
    etf.appendChild(el("label", "cap", "Max tokens — pick a size (low → high VRAM)"));
    this.enhTok = el("select");
    this.enhTok.onchange = () => { this.setW("llm_max_token", this.enhTok.value); this.enhTok.title = this.enhTok.value; };
    etf.appendChild(this.enhTok); this.enhCfg.appendChild(etf);
    this.enhSec.appendChild(this.enhCfg);
    L.appendChild(this.enhSec);   // Prompt Enhancer on the LEFT column (below the LoRA stack)



    // save name / path — YOU decide it. Used verbatim as SaveImage's filename_prefix
    // (folder + filename), unless auto-organize is on. save_root is global (shared).
    this.saveSec = el("div", "sec");
    this.saveSec.appendChild(el("label", "cap", "Save name / path — you decide it"));
    const sr = el("div", "cmdrow");
    this.saveRoot = el("input"); this.saveRoot.type = "text";
    this.saveRoot.placeholder = "e.g.  MyProject/portrait   (folder + filename)";
    this.saveRoot.title = "Used exactly as you type it, under ComfyUI/output. Include folders "
      + "with / and a filename — e.g. MyProject/portrait -> output/MyProject/portrait_00001_.png";
    this.saveRoot.oninput = () => { this.setW("save_root", this.saveRoot.value); this.refreshSavePath(); };
    sr.appendChild(this.saveRoot);
    this.saveSec.appendChild(sr);
    // optional: let the node auto-file per pipeline instead (off by default)
    const aol = el("label", "chk"); aol.style.marginTop = "6px";
    this.autoOrg = el("input"); this.autoOrg.type = "checkbox";
    this.autoOrg.onchange = () => { this.setW("auto_organize", this.autoOrg.checked); this.refreshSavePath(); };
    aol.appendChild(this.autoOrg);
    aol.appendChild(el("span", null, "Auto-organize into per-pipeline subfolders"));
    this.saveSec.appendChild(aol);
    this.saveSec.appendChild(el("div", "muted", "Where the next file lands (your text decides it):"));
    this.savePreview = el("div", "savepath");
    this.saveSec.appendChild(this.savePreview);
    R.appendChild(this.saveSec);   // Save sits directly under the Live preview on the RIGHT

    // optional extras + LoRAs
    this.depBox = el("button", "refbtn plainbtn");
    this.depBox.appendChild(el("span", null, "Optional extras & LoRA downloads"));
    const depBody = el("div");
    this.depBox.onclick = () => this.openOverlay("Optional extras & LoRAs", depBody);
    depBody.appendChild(el("div", "muted",
      "The core edit / identity / inpaint / outpaint nodes are BUNDLED in this pack — " +
      "nothing else is needed to run. The packs below are OPTIONAL, each only for its one feature."));
    for (const [pack, used, url] of DEPENDENCIES) {
      const row = el("div", "deprow");
      if (url) {
        // Link to the repo PAGE (not a zip) so users can git clone it.
        const a = el("a", "deplink", pack);
        a.href = url;
        a.target = "_blank"; a.rel = "noopener";
        a.title = "Open " + url + " — git clone it into ComfyUI/custom_nodes";
        row.appendChild(a);
      } else {
        row.appendChild(el("span", "deplink builtin", pack));
      }
      row.appendChild(el("span", "depuse", used));
      depBody.appendChild(row);
    }
    depBody.appendChild(el("div", "muted",
      "Only if you want that optional feature: open the link and git clone it into " +
      "ComfyUI/custom_nodes/, then restart ComfyUI.  e.g.  git clone <link>.git"));
    for (const [name, used, url] of LORAS) {
      const row = el("div", "deprow");
      const a = el("a", "deplink", name);
      a.href = url; a.target = "_blank"; a.rel = "noopener";
      a.title = url;
      row.appendChild(a);
      row.appendChild(el("span", "depuse", used));
      depBody.appendChild(row);
    }
    depBody.appendChild(el("div", "muted", "LoRAs go in ComfyUI/models/loras/Krea2/."));

    // Sharing this node: a LAN clone URL is useless to anyone not on this network,
    // so point at the zip, which works for everybody.
    depBody.appendChild(el("div", "muted", "SHARING THIS NODE"));
    depBody.appendChild(el("div", "sharenote",
      "Send the whole KreaUltraController folder (or Krea2_AIO_AJ.zip) — drop it in " +
      "ComfyUI/custom_nodes/ and restart. The links above cover everything else it needs."));

    this.foot.appendChild(this.depBox);

    M.appendChild(this.foot);   // Optional extras & LoRA downloads sit under Remove background (middle)
    this.status = el("div", "status");
    r.appendChild(this.status);
  }

  // Is a named input socket actually connected?
  connected(name) {
    const inp = (this.node.inputs || []).find((i) => i.name === name);
    return !!(inp && inp.link != null);
  }

  renderConnections() {
    const p = this.pipe();
    const outp = String(this.gv("fill_mode") || "").startsWith("B");
    this.connList.innerHTML = "";
    const rows = SOCKETS[p] || [];
    if (!rows.length) {
      this.connList.appendChild(el("div", "muted", "This pipeline needs no image input."));
      return;
    }
    for (const [name, req] of rows) {
      // mask is required only for INPAINT; outpaint builds its own
      const needed = req === "inpaint" ? (p === 3 && !outp) : req;
      if (name === "mask" && p === 3 && outp) continue;
      let ok = this.connected(name);
      const ignored = (name === "reference") && ok && this.gv("use_reference") === false;
      const row = el("div", "lrow");
      if (!ok) row.classList.add("off");
      const dot = el("span", null, ok ? "●" : "○");
      dot.style.cssText = `flex:none;font-size:11px;color:${ok ? "#5a9c5a" : (needed ? "#e0a33e" : "#666")}`;
      const label = el("span", "lname", name);
      const hint = el("span", null,
        ignored ? "connected · IGNORED" : (ok ? "connected" : (needed ? "REQUIRED" : "optional")));
      hint.style.cssText = "flex:none;font-size:9.5px;opacity:.7";
      if (!ok && needed) hint.style.color = "#e0a33e";
      if (ignored) { hint.style.color = "#e0a33e"; row.classList.add("off"); }
      row.appendChild(dot); row.appendChild(label); row.appendChild(hint);
      this.connList.appendChild(row);
    }
    if (p === 3 && !outp) {
      this.connList.appendChild(el("div", "muted",
        "Paint the mask on the LoadImage node (right-click → Open in MaskEditor). " +
        "It is the alpha channel — do not paint green yourself."));
    } else if (p === 2) {
      this.connList.appendChild(el("div", "muted",
        "mask is optional here: it becomes edit_mask on the native-res restore, so only " +
        "what you painted is composited back. Leave it unwired for a normal edit."));
    }
  }

  mkSlot(widgetName, title, optional) {
    const row = el("div", "slot");
    const img = el("img", "thumb");
    img.onerror = () => { img.style.visibility = "hidden"; };
    const body = el("div", "slotbody");
    body.appendChild(el("div", "slotttl", title));
    const sel = el("select");
    const btns = el("div", "btnrow");
    const up = el("button", null, "Upload");
    const me = el("button", null, "Mask editor");
    btns.appendChild(up); btns.appendChild(me);
    body.appendChild(sel); body.appendChild(btns);
    row.appendChild(img); row.appendChild(body);

    const slot = { row, img, sel, up, me, widgetName, optional, filled: false };
    sel.onchange = () => {
      this.setW(widgetName, sel.value);
      img.src = thumbUrl(sel.value); img.style.visibility = "";
    };
    up.onclick = () => this.upload(slot);
    me.onclick = () => this.maskEditor(slot);
    return slot;
  }

  fillSlot(slot) {
    const w = this.w(slot.widgetName);
    if (!w) return;
    const opts = w.options?.values || [];
    const list = typeof opts === "function" ? opts(w) : opts;
    if (slot.sel.dataset.n !== String(list.length)) {
      slot.sel.innerHTML = "";
      for (const o of list) { const op = el("option", null, o); op.value = o; slot.sel.appendChild(op); }
      slot.sel.dataset.n = String(list.length);
    }
    slot.sel.value = w.value;
    slot.img.src = thumbUrl(w.value);
    slot.img.style.visibility = "";
  }

  async upload(slot) {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = "image/*";
    inp.onchange = async () => {
      const f = inp.files?.[0]; if (!f) return;
      const fd = new FormData();
      fd.append("image", f); fd.append("overwrite", "false");
      try {
        const r = await api.fetchApi("/upload/image", { method: "POST", body: fd });
        if (!r.ok) throw new Error("HTTP " + r.status);
        const d = await r.json();
        const name = d.subfolder ? `${d.subfolder}/${d.name}` : d.name;
        const w = this.w(slot.widgetName);
        const opts = w?.options?.values;
        if (Array.isArray(opts) && !opts.includes(name)) opts.push(name);
        const op = el("option", null, name); op.value = name;
        slot.sel.appendChild(op); slot.sel.value = name;
        this.setW(slot.widgetName, name);
        slot.img.src = thumbUrl(name); slot.img.style.visibility = "";
      } catch (e) {
        console.error("[KreaAIO] upload failed:", e);
        alert("Upload failed — see console.");
      }
    };
    inp.click();
  }

  // The mask editor works on a node's image preview. We temporarily point the node's
  // imgs at the chosen file so the standard editor opens on it and writes a clipspace
  // file back, exactly as it does for LoadImage.
  async maskEditor(slot) {
    const w = this.w(slot.widgetName);
    if (!w || !w.value || w.value === "None") { alert("Pick an image first."); return; }
    try {
      const im = new Image();
      im.src = thumbUrl(w.value);
      await new Promise((res, rej) => { im.onload = res; im.onerror = rej; });
      this.node.imgs = [im];
      this.node.imageIndex = 0;
      this.node._kaioMaskTarget = slot.widgetName;
      app.canvas.selectNode(this.node);
      await app.extensionManager.command.execute("Comfy.MaskEditor.OpenMaskEditor");
    } catch (e) {
      console.error("[KreaAIO] mask editor failed:", e);
      alert("Could not open the mask editor — see console.");
    }
  }

  // ---- loras -------------------------------------------------------------
  loras() {
    try {
      const v = JSON.parse(this.gv("loras_json") || "[]");
      return Array.isArray(v) ? v : [];
    } catch (e) { return []; }
  }
  saveLoras(list) {
    const json = JSON.stringify(list);
    this.setW("loras_json", json);           // active stack the backend reads
    const s = this.state();
    if (!s._loras) s._loras = {};
    s._loras[this.loraKey()] = json;         // persist to the current slot
    this.saveState(s);
  }

  togglePicker() {
    this.picker.classList.toggle("hide");
    if (!this.picker.classList.contains("hide")) {
      this.pickInput.value = "";
      this.renderPicker();
      this.pickInput.focus();
    }
  }

  async renderPicker() {
    const all = await loraList();
    const q = this.pickInput.value.toLowerCase().trim();
    const hits = (q ? all.filter((x) => x.toLowerCase().includes(q)) : all).slice(0, 200);
    this.pickList.innerHTML = "";
    if (!hits.length) { this.pickList.appendChild(el("div", "muted", "no matches")); return; }
    for (const x of hits) {
      const it = el("div", "pickitem", baseName(x));
      it.title = x;
      it.onclick = () => {
        const list = this.loras();
        list.push({ on: true, lora: x, strength: 1.0 });
        this.saveLoras(list);
        this.picker.classList.add("hide");
        this.renderLoras();
        // convenience: pull this LoRA's trigger words into the shared trigger box
        this.fetchTriggers(x).then((t) => {
          t = (t || "").trim();
          if (!t) return;
          const cur = (this.gv("trigger_words") || "").trim();
          if (cur.toLowerCase().includes(t.toLowerCase())) return;
          const merged = cur ? cur + ", " + t : t;
          this.setP("trigger_words", merged);
          if (this.trigWords) this.trigWords.value = merged;
        });
      };
      this.pickList.appendChild(it);
    }
  }

  renderLoras() {
    const list = this.loras();
    const isB = String(this.gv("edit_mode") || "").startsWith("B");
    const p = this.pipe();
    this.lList.innerHTML = "";
    if (!list.length) {
      this.lList.appendChild(el("div", "muted", "No LoRAs. Add one below."));
    }
    list.forEach((entry, i) => {
      const row = el("div", "lrow");
      if (!entry.on) row.classList.add("off");
      const cb = el("input"); cb.type = "checkbox"; cb.checked = !!entry.on;
      cb.onchange = () => { const l = this.loras(); l[i].on = cb.checked; this.saveLoras(l); this.renderLoras(); this.updateStatus(); };
      const nm = el("span", "lname", baseName(entry.lora));
      nm.title = entry.lora;
      if (p === 2 && entry.on) {
        const b = baseName(entry.lora);
        const okA = b === MODE_A_LORA;
        const okB = MODE_B_LORAS.some((x) => b.toLowerCase().includes(x.toLowerCase()));
        if ((!isB && !okA) || (isB && !okB)) nm.classList.add("bad");
      }
      const st = el("input"); st.type = "number"; st.className = "lstr"; st.step = 0.05;
      st.value = entry.strength ?? 1;
      st.oninput = () => { const l = this.loras(); l[i].strength = Number(st.value || 0); this.saveLoras(l); };
      // LoRAs are applied in list order and each one patches the model the previous one
      // produced, so the order genuinely changes the result. These move a LoRA up or down
      // the chain. Position 1 is applied first.
      const mv = (dir) => {
        const l = this.loras();
        const j = i + dir;
        if (j < 0 || j >= l.length) return;
        [l[i], l[j]] = [l[j], l[i]];
        this.saveLoras(l); this.renderLoras(); this.updateStatus();
      };
      const up = el("button", "lmv", "▲");
      up.title = "Move earlier in the chain (applied sooner)";
      up.disabled = i === 0;
      up.onclick = () => mv(-1);
      const dn = el("button", "lmv", "▼");
      dn.title = "Move later in the chain (applied after the ones above)";
      dn.disabled = i === list.length - 1;
      dn.onclick = () => mv(1);
      const ord = el("span", "lord", String(i + 1));
      ord.title = `Applied ${i === 0 ? "first" : i === list.length - 1 ? "last" : "#" + (i + 1)} in the chain`;
      const x = el("button", "lx", "×");
      x.onclick = () => { const l = this.loras(); l.splice(i, 1); this.saveLoras(l); this.renderLoras(); this.updateStatus(); };
      row.appendChild(ord); row.appendChild(cb); row.appendChild(nm);
      row.appendChild(st); row.appendChild(up); row.appendChild(dn); row.appendChild(x);
      this.lList.appendChild(row);
    });
  }

  // Best-effort read of a LoRA's trigger/trained words from its safetensors metadata
  // (served by the /kaio/lora_triggers route in server_routes.py). Empty on failure.
  async fetchTriggers(name) {
    try {
      const r = await fetch("/kaio/lora_triggers?name=" + encodeURIComponent(name));
      if (!r.ok) return "";
      const d = await r.json();
      return (d && d.triggers) || "";
    } catch (e) { return ""; }
  }

  // ---- embedded live preview --------------------------------------------
  // ComfyUI streams sampler preview frames as `b_preview` events (a Blob each) and
  // `progress` events. We paint them into the fixed box; nothing here changes the node
  // size, so a running generation never disturbs the layout.
  setupLivePreview() {
    this._prevFrames = {};   // last frame per pipeline, so switching shows THAT workflow's
    const showBlob = (blob) => {
      if (!(blob instanceof Blob)) return;
      const p = this.pipe();
      if (this._prevFrames[p]) URL.revokeObjectURL(this._prevFrames[p]);
      const url = URL.createObjectURL(blob);
      this._prevFrames[p] = url;
      this.prevImg.src = url;
      this.prevImg.style.display = "block";
      this.prevHint.style.display = "none";
    };
    this._showBlob = showBlob;
    // Old preview path: detail IS the Blob. New path (1.4x): detail is {blob, ...}.
    // Listen to BOTH so it works regardless of which the runtime uses.
    this._prevCb = (e) => showBlob(e.detail instanceof Blob ? e.detail : (e.detail && e.detail.blob));
    this._prevMetaCb = (e) => showBlob(e.detail && e.detail.blob);
    this._progCb = (e) => {
      const d = e.detail || {};
      const mx = Number(d.max) || 0, v = Number(d.value) || 0;
      const pct = mx ? Math.min(100, Math.round(v / mx * 100)) : 0;
      this.prevBar.style.width = pct + "%";
      this.prevPct.textContent = pct + "%";
      this.prevPct.style.display = (pct > 0 && pct < 100) ? "" : "none";
      if (pct >= 100) setTimeout(() => { this.prevBar.style.width = "0%"; this.prevPct.style.display = "none"; }, 500);
    };
    api.addEventListener("b_preview", this._prevCb);
    api.addEventListener("b_preview_with_metadata", this._prevMetaCb);
    api.addEventListener("progress", this._progCb);
    this.setupDoneChime();
  }

  // The chime is triggered from the node's onExecuted() hook (see beforeRegisterNodeDef),
  // which is the same mechanism pysssss's PlaySound uses and the only one that reliably
  // fires on this frontend. onExecuted only runs if the backend returns a `ui` payload,
  // which controller.py now does ({"kaio_done": …}) for exactly this purpose.
  setupDoneChime() { /* wiring lives in onExecuted; nothing to bind here */ }

  chime() {
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      const ctx = new AC();
      if (ctx.state === "suspended") ctx.resume().catch(() => { });
      const vol = 0.5;
      // Two soft notes (C6 -> E6): pleasant, short, hard to mistake for a system error.
      [[1046.5, 0], [1318.5, 0.16]].forEach(([hz, at]) => {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.type = "sine"; o.frequency.value = hz;
        const t = ctx.currentTime + at;
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(Math.max(0.0002, vol * 0.6), t + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 0.45);
        o.connect(g); g.connect(ctx.destination);
        o.start(t); o.stop(t + 0.5);
      });
      setTimeout(() => { try { ctx.close(); } catch (e) { } }, 1200);
    } catch (e) { console.warn("[KreaAIO] chime failed:", e); }
  }

  destroyPreview() {
    try {
      if (this._prevCb) api.removeEventListener("b_preview", this._prevCb);
      if (this._prevMetaCb) api.removeEventListener("b_preview_with_metadata", this._prevMetaCb);
      if (this._progCb) api.removeEventListener("progress", this._progCb);
      if (this._ro) { this._ro.disconnect(); this._ro = null; }
    } catch (e) { /* ignore */ }
  }

  // Show the preview stored for the CURRENT workflow (each pipeline keeps its own last
  // frame), or the hint if that workflow hasn't generated yet.
  showStoredPreview() {
    if (!this.prevImg) return;
    const u = this._prevFrames && this._prevFrames[this.pipe()];
    if (u) { this.prevImg.src = u; this.prevImg.style.display = "block"; this.prevHint.style.display = "none"; }
    else { this.prevImg.removeAttribute("src"); this.prevImg.style.display = "none"; this.prevHint.style.display = ""; }
  }

  // ---- sync --------------------------------------------------------------
  fillCombo(sel, wname) {
    const w = this.w(wname);
    if (!w) return;
    const opts = w.options?.values || [];
    const list = typeof opts === "function" ? opts(w) : opts;
    if (sel.dataset.n !== String(list.length)) {
      sel.innerHTML = "";
      for (const o of list) {
        const op = el("option", null, String(o).split(/[\\/]/).pop());
        op.value = o; op.title = o;
        sel.appendChild(op);
      }
      sel.dataset.n = String(list.length);
    }
    sel.value = w.value; sel.title = String(w.value);
  }

  sync() {
    const p = this.pipe();
    const up = this.upscale();
    const shown = new Set(SHOW[p] || []);
    if (up) for (const k of UPSCALE_EXTRA) shown.add(k);

    for (const q of PIPES) this.tabEl[q.idx].classList.toggle("on", q.idx === p);
    this.upBtn.classList.toggle("on", up);

    // contextual guide at the TOP of the panel — its contents follow the selected
    // workflow (the upscale layer, key 5, shows its own short guidance).
    const key = up ? 5 : p;
    const [title, rows] = NOTES[key] || NOTES[4];
    this.guideTitle.textContent = "How to use — " + title;
    this.guideBody.innerHTML = "";
    const dl = el("dl");
    for (const [a, b] of rows) { dl.appendChild(el("dt", null, a)); dl.appendChild(el("dd", null, b)); }
    this.guideBody.appendChild(dl);
    // If the guide bubble is currently open, refresh it in place for the new workflow.
    if (this._guideOpen && !this.ovl.classList.contains("hide")) {
      this.openOverlay(this.guideTitle.textContent, this.guideBody);
    }

    this.prompt.value = this.gv("prompt") ?? "";
    if (this.negPrompt) {
      this.negPrompt.value = this.gv("negative_prompt") ?? "";
      const cfg = Number(this.gv("cfg") ?? 1);
      const live = cfg > 1.05;
      this.negSum.textContent = live
        ? "Negative prompt — ACTIVE (CFG " + cfg + ")"
        : "Negative prompt — inactive at CFG " + cfg;
      this.negHint.textContent = live
        ? "Active. Describe what the model keeps adding unasked."
        : "Krea 2 Turbo runs at CFG 1.0, where a negative prompt does nothing at all. "
        + "To use it, load a RAW checkpoint and raise CFG to about 5.5 with ~28 steps.";
      this.negHint.style.color = live ? "#8fbf8f" : "var(--warn)";
    }
    if (this.trigWords) this.trigWords.value = this.gv("trigger_words") ?? "";
    if (this.seedMode) this.seedMode.value = this.gv("control_after_generate") ?? "randomize";
    for (const k of Object.keys(this.num)) {
      this.num[k].i.value = this.gv(k) ?? 0;
      this.num[k].f.classList.toggle("hide", !shown.has(k));
    }
    for (const k of Object.keys(this.combo)) {
      this.fillCombo(this.combo[k].s, k);
      this.combo[k].f.classList.toggle("hide", !shown.has(k));
    }
    this.fillCombo(this.restore, "restore_mode");
    for (const [k, s] of Object.entries(this.modelRows)) this.fillCombo(s, k);

    this.modeSec.classList.toggle("hide", !shown.has("edit_mode"));
    this.fillSec.classList.toggle("hide", !shown.has("fill_mode"));
    this.rSec.classList.toggle("hide", !shown.has("restore_mode"));
    this.faceSec.classList.toggle("hide", !shown.has("face_detail"));
    this.rbSec.classList.toggle("hide", !shown.has("remove_background"));
    this.imgSec.classList.toggle("hide", !shown.has("conn"));
    this.lSec.classList.toggle("hide", !shown.has("loras"));

    const isB = String(this.gv("edit_mode") || "").startsWith("B");
    this.mA.classList.toggle("on", !isB);
    this.mB.classList.toggle("on", isB);
    // MODE B routes through a different reference path — these are MODE A dials.
    if (p === 2) {
      this.num.grounding_px.f.classList.toggle("hide", isB);
      this.num.ref_boost.f.classList.toggle("hide", isB);
    }

    // Hide a whole labelled group when none of its fields apply, so there are no empty
    // captioned boxes. Sampler always applies.
    // Output-size mode (pipelines 1 & 2). When it's on "match input image size" the aspect and
    // megapixel fields do nothing, so hide them rather than leave dead controls on screen.
    const sizeVis = shown.has("size_mode");
    if (this.sizeSec) this.sizeSec.classList.toggle("hide", !sizeVis);
    if (this.sizeSel) this.fillCombo(this.sizeSel, "size_mode");
    const matchInput = sizeVis && !String(this.gv("size_mode") || "").toLowerCase().startsWith("aspect");
    if (matchInput) { shown.delete("aspect_ratio"); shown.delete("megapixels"); }
    for (const k of ["aspect_ratio", "megapixels"]) {
      if (this.num[k]) this.num[k].f.classList.toggle("hide", !shown.has(k));
      if (this.combo[k]) this.combo[k].f.classList.toggle("hide", !shown.has(k));
    }

    const resVis = sizeVis || shown.has("aspect_ratio") || shown.has("megapixels");
    const refVis = (shown.has("grounding_px") || shown.has("ref_boost")) && !(p === 2 && isB);
    this.resSec.classList.toggle("hide", !resVis);
    this.refSec.classList.toggle("hide", !refVis);

    const outp = String(this.gv("fill_mode") || "").startsWith("B");
    this.fA.classList.toggle("on", !outp);
    this.fB.classList.toggle("on", outp);

    this.renderConnections();
    this.useRef.checked = this.gv("use_reference") !== false;
    this.refWrap.classList.toggle("hide", !shown.has("use_reference"));
    this.face.checked = !!this.gv("face_detail");
    this.rb.checked = !!this.gv("remove_background");

    // prompt enhancer — the section is always visible; the model + token controls show
    // only once it's enabled. enhancer_model / llm_max_token are global (setW).
    if (this.enhChk) this.enhChk.checked = !!this.gv("enhance_prompt");
    if (this.enhModel) this.fillCombo(this.enhModel, "enhancer_model");
    if (this.enhTok) this.fillCombo(this.enhTok, "llm_max_token");
    if (this.enhCfg) this.enhCfg.classList.toggle("hide", !this.gv("enhance_prompt"));

    // Flux loaders appear with the upscale; outpaint padding with pipeline 3 OUTPAINT.
    this.fluxWrap.classList.toggle("hide", !up);
    this.upSteps.value = this.gv("upscale_steps") ?? 2;
    this.upMP.value = this.gv("upscale_megapixels") ?? 4;

    // RTX VSR: the tick is always available, its dials follow the tick. Global (setW),
    // not per-pipeline — it is a hardware capability, not a creative setting.
    if (this.vsrChk) {
      const vsrOn = !!this.gv("rtx_vsr");
      this.vsrChk.checked = vsrOn;
      this.vsrScale.value = this.gv("rtx_vsr_scale") ?? 2;
      this.fillCombo(this.vsrQual, "rtx_vsr_quality");
      this.vsrCfg.classList.toggle("hide", !vsrOn);
    }
    const showPad = (p === 3) && outp;
    this.padSec.classList.toggle("hide", !showPad);
    for (const k of Object.keys(this.pad)) this.pad[k].i.value = this.gv(k) ?? 0;
    if (up && !this.mBox.open) this.mBox.open = true;

    this.saveRoot.value = this.gv("save_root") ?? "Krea2AJ";
    if (this.autoOrg) this.autoOrg.checked = !!this.gv("auto_organize");
    this.refreshSavePath();
    this.renderLoras();
    this.updateStatus();
    this.showStoredPreview();   // this workflow's own last preview
    this.remeasure();
  }

  remeasure() {
    this.applyHeight();
  }

  openOverlay(title, contentEl) {
    this.ovlTitle.textContent = title;
    this.ovlBody.innerHTML = "";
    this.ovlBody.appendChild(contentEl);
    this.ovl.classList.remove("hide");
  }

  // Per-workflow explanation of the LLM prompt enhancer (opened by the "i" in its section).
  buildEnhInfo() {
    const wrap = el("div");
    const dl = el("dl");
    const add = (t, d) => { dl.appendChild(el("dt", null, t)); dl.appendChild(el("dd", null, d)); };
    add("What it does",
      "Rewrites your prompt with a small LLM (the Krea 2 text encoder you already load) BEFORE " +
      "generating — the same idea as ComfyUI's official Krea-2 workflow. Your LoRA trigger words " +
      "are kept exactly and appended AFTER the rewrite, so they are never lost.");
    add("For this workflow", ENH_INFO[this.pipe()] || ENH_INFO[4]);
    add("Text encoder (enhancer) + VRAM",
      "The default '(loaded text encoder)' reuses the Qwen3-VL you already loaded — no extra VRAM. " +
      "Choosing a different text encoder loads THAT one just for the rewrite (heavier). Use an " +
      "abliterated Qwen3-VL here for no refusals.");
    add("Max tokens (VRAM)",
      "How long the rewritten prompt can get. 256 is the safe LOW-VRAM default. Higher values " +
      "(up to 2048) give longer, richer prompts but use more VRAM and take longer to generate — " +
      "raise it only on a higher-VRAM card.");
    add("Spicy / NSFW prompts",
      "The built-in instructions do not sanitise, cover up, or refuse what you wrote. But the base " +
      "Qwen3-VL still has its own safety training and may refuse anyway — for fully uncensored " +
      "expansion, pick an ABLITERATED Qwen3-VL in the model dropdown.");
    add("If it fails",
      "Any error, empty result, or refusal falls back to your prompt exactly as typed, so a run is " +
      "never blocked by the enhancer.");
    wrap.appendChild(dl);
    return wrap;
  }

  refreshSavePath() {
    const p = this.pipe();
    let path;
    if (this.gv("auto_organize")) {
      path = buildSavePath(
        this.gv("save_root"), p,
        String(this.gv("edit_mode") || "").startsWith("B"),
        String(this.gv("fill_mode") || "").startsWith("B"),
        !!this.gv("face_detail"), this.upscale());
    } else {
      // used exactly as typed (folder + filename)
      path = String(this.gv("save_root") || "Krea2AJ").trim().replace(/^[\/\\]+|[\/\\]+$/g, "") || "Krea2AJ";
    }
    this.savePreview.textContent = "output/" + path + "_00001_.png";
    const wired = (this.node.outputs || []).some(
      (o) => o.name === "save_path" && (o.links || []).length);
    this.savePreview.classList.toggle("unwired", !wired);
    this.savePreview.title = wired
      ? "save_path is driving your SaveImage node — your typed name/path decides the location."
      : "Connect the save_path output to SaveImage.filename_prefix to use it, or set SaveImage's prefix yourself.";
  }

  updateStatus() {
    const p = this.pipe(), up = this.upscale();
    const warns = [];
    const isB = String(this.gv("edit_mode") || "").startsWith("B");
    if (p === 2) {
      for (const e of this.loras()) {
        if (!e.on) continue;
        const b = baseName(e.lora);
        const okA = b === MODE_A_LORA;
        const okB = MODE_B_LORAS.some((x) => b.toLowerCase().includes(x.toLowerCase()));
        if (!isB && !okA) warns.push(`MODE A expects ${MODE_A_LORA}; "${b}" is on.`);
        if (isB && !okB) warns.push(`MODE B takes ai-toolkit LoRAs only; "${b}" is on.`);
      }
    }
    const outp = String(this.gv("fill_mode") || "").startsWith("B");
    for (const [name, req] of SOCKETS[p] || []) {
      const needed = req === "inpaint" ? (p === 3 && !outp) : req;
      if (needed && !this.connected(name)) {
        warns.push(`\`${name}\` is not connected — this pipeline needs it.`);
      }
    }

    this.status.innerHTML = "";
    // Removed the always-on "Active: … / N LoRAs active" lines — redundant with the highlighted
    // tab and the LoRA list, and they added height and an odd empty bar. Only real warnings
    // appear here now.
    for (const wn of warns) this.status.appendChild(el("div", "warn", "! " + wn));

    // Optional-feature packs: if the user enabled a feature but its pack isn't installed,
    // the run still works (that step is skipped) — but tell them, with a download link.
    const OPT = [
      { on: !!this.gv("remove_background") && p === 1, type: "easy imageRemBg",
        feat: "Background removal", pack: "ComfyUI-Easy-Use", url: "https://github.com/yolain/ComfyUI-Easy-Use" },
      { on: !!this.gv("face_detail") && (p === 1 || p === 4), type: "FaceDetailer",
        feat: "Face detail", pack: "ComfyUI-Impact-Pack", url: "https://github.com/ltdrdata/ComfyUI-Impact-Pack" },
      { on: up, type: "ColorMatch",
        feat: "Upscale colour match", pack: "ComfyUI-KJNodes", url: "https://github.com/kijai/ComfyUI-KJNodes" },
    ];
    for (const o of OPT) {
      if (!o.on || !this.nodeMissing(o.type)) continue;
      const d = el("div", "warn");
      d.appendChild(document.createTextNode(`! ${o.feat} needs ${o.pack} (not installed — step is skipped). `));
      const a = el("a", "deplink", "git clone →");
      a.href = o.url; a.target = "_blank"; a.rel = "noopener"; a.title = o.url;
      d.appendChild(a);
      this.status.appendChild(d);
    }
    // Hide the status box entirely when there's nothing to warn about, so it never shows as an
    // empty bar at the bottom (no bloat, and the node auto-fit shrinks accordingly).
    this.status.classList.toggle("hide", !this.status.children.length);
  }

  // True if a node type is NOT registered in the frontend (i.e. its pack isn't installed).
  // Unknown/registry-unavailable -> false, so we never warn on a false positive.
  nodeMissing(type) {
    try {
      const reg = window.LiteGraph && window.LiteGraph.registered_node_types;
      if (!reg) return false;
      return !(reg[type] || reg[type.toLowerCase()]);
    } catch (e) { return false; }
  }
}

// Map each model dropdown's value to a file that actually exists on THIS machine, by
// filename. Fixes the false "Missing Models" nag when the saved value uses a different
// folder layout (e.g. basename vs "Krea2\\file"). Flux loaders fall back to the first
// available model when there's no match, so they never nag while the upscaler is unused.
const MODEL_WIDGETS = ["unet_name", "clip_name", "vae_name",
                       "flux_unet_name", "flux_clip_name", "flux_vae_name"];
function resolveModelWidgets(node) {
  const base = (s) => String(s || "").split(/[\\/]/).pop().toLowerCase();
  for (const key of MODEL_WIDGETS) {
    const w = (node.widgets || []).find((x) => x.name === key);
    if (!w) continue;
    const opts = w.options && w.options.values;
    const list = typeof opts === "function" ? opts(w) : (opts || []);
    if (!Array.isArray(list) || !list.length) continue;
    if (list.includes(w.value)) continue;                 // already valid
    const m = list.find((o) => base(o) === base(w.value)); // same filename, other folder
    if (m) { w.value = m; if (w.callback) { try { w.callback(m); } catch (e) { } } continue; }
    if (key.startsWith("flux_")) {                         // unused-upscaler: keep it valid
      w.value = list[0]; if (w.callback) { try { w.callback(list[0]); } catch (e) { } }
    }
  }
}

function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  if (!w.options) w.options = {};
  w.options.hidden = true;
  if (!window.LiteGraph || !window.LiteGraph.vueNodesMode) {
    w.computeSize = () => [0, -4];
    w.draw = () => { };
  }
  if (w.element) w.element.style.display = "none";
}

app.registerExtension({
  name: "KreaAIO",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "KreaAIO") return;

    // LiteGraph's computeSize() short-circuits on the class's static size, so this
    // makes EVERY caller get the fixed size — including ComfyUI's own re-layout paths.
    nodeType.size = [NODE_W, NODE_H];

    // During sampling ComfyUI attaches a live preview to the executing node and calls
    // setSizeForImage() to refit the node around it, which squashes this UI to ~200px
    // wide. This node has a DOM body, not an image body, so that refit is never
    // wanted. Results are viewed through the SaveImage / Image Comparer nodes.
    nodeType.prototype.setSizeForImage = function () { };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      if (onNodeCreated) onNodeCreated.apply(this, arguments);
      for (const w of this.widgets || []) hideWidget(w);

      const c = document.createElement("div");
      c.style.boxSizing = "border-box";
      // Own the width from the moment the element exists, not just from when AIO is
      // constructed (that happens a tick later). See AIO.applyWidth() for why.
      c.style.width = c.style.minWidth = (NODE_W - 26) + "px";
      const widget = this.addDOMWidget("kaio_ui", "kaio_ui", c, {
        getValue: () => "", setValue: () => { },
      });
      const self = this;
      // Fixed size (NOT user-resizable). A constant computeSize can never feed back into
      // the node size, so there is no growth loop. The node is sized large enough that
      // every pipeline's controls fit without scrolling.
      widget.computeSize = function () {
        return [NODE_W - 26, NODE_H - 64];
      };
      this.resizable = false;   // user cannot resize; the dev-set size is authoritative
      this.size[0] = NODE_W;
      this.size[1] = NODE_H;
      resolveModelWidgets(this);   // map model values to real files so no false "missing"

      // Swallow any preview images assigned during execution: keeping imgs empty stops
      // the image-layout path entirely, DOM widget stays the whole body.
      Object.defineProperty(this, "imgs", {
        get() { return this._kaioImgs; },
        set(v) { this._kaioImgs = undefined; },
        configurable: true,
      });

      setTimeout(() => {
        try { self._kaio = new AIO(self, c); }
        catch (err) {
          console.error("[KreaAIO] UI init failed:", err);
          c.textContent = "KREA2 AJ AIO UI failed to load — see console.";
        }
      }, 0);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const out = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      const self = this;
      // Fixed size; never restore one from the file. Resolve model values BEFORE the
      // frontend's missing-models scan runs so it doesn't falsely flag them.
      this.size[0] = NODE_W; this.size[1] = NODE_H;
      resolveModelWidgets(this);
      setTimeout(() => { try { self._kaio?.migrateOrInit(); self._kaio?.loadLorasForCurrent(); self._kaio?.sync(); } catch (e) { } }, 48);
      return out;
    };

    // Keep the input readout honest when links are made or broken.
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const out = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
      const self = this;
      setTimeout(() => {
        try { self._kaio?.renderConnections(); self._kaio?.updateStatus(); } catch (e) { }
      }, 0);
      return out;
    };

    // Finish chime. onExecuted fires once the backend returns its ui payload, i.e. after the
    // WHOLE node is done — pipeline, face detail, Flux 2 upscale and save included. This is
    // the same hook pysssss's PlaySound uses, and unlike the websocket events it fires
    // reliably on every frontend version.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      const out = onExecuted ? onExecuted.apply(this, arguments) : undefined;
      try {
        if (output && output.kaio_done) this._kaio?.chime();
      } catch (e) { console.warn("[KreaAIO] chime failed:", e); }
      return out;
    };

    // Detach the live-preview websocket listeners when the node is removed.
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      try { this._kaio?.destroyPreview(); } catch (e) { }
      return onRemoved ? onRemoved.apply(this, arguments) : undefined;
    };
  },
});
