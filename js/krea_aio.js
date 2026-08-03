// KREA2 AJ AIO — embedded UI.
//
// The node runs all five pipelines internally (see pipelines.py), so this file does no
// graph manipulation at all. It hides the node's real widgets and renders a compact
// control surface that writes back into them, which is what reaches the backend.

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

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
  1: ["conn", "prompt", "seed", "steps", "cfg", "sampler", "scheduler",
      "aspect_ratio", "megapixels", "grounding_px", "ref_boost", "loras",
      "face_detail", "remove_background", "use_reference"],
  2: ["conn", "prompt", "seed", "steps", "cfg", "sampler", "scheduler",
      "grounding_px", "ref_boost", "restore_mode", "loras", "edit_mode",
      "use_reference"],
  3: ["conn", "prompt", "seed", "steps", "cfg", "sampler", "scheduler", "loras", "fill_mode"],
  4: ["prompt", "seed", "steps", "cfg", "sampler", "scheduler", "aspect_ratio", "megapixels",
      "loras", "face_detail"],
};
const UPSCALE_EXTRA = ["megapixels"];

// Which sockets each pipeline actually consumes. `mask` is INPAINT-only by design:
// outpaint generates its own mask, and pipelines 1/2/4 ignore it entirely.
const SOCKETS = {
  1: [["image", true], ["reference", false]],
  2: [["image", true], ["reference", false]],
  3: [["image", true], ["mask", "inpaint"]],
  4: [],
};

const PRESETS = {
  1: { steps: 10, cfg: 1.0, megapixels: 2.0, grounding_px: 512, ref_boost: 4.0 },
  2: { steps: 8, cfg: 1.0, megapixels: 1.0, grounding_px: 768, ref_boost: 1.0 },
  3: { steps: 8, cfg: 1.0, megapixels: 1.0, grounding_px: 768, ref_boost: 1.0 },
  4: { steps: 8, cfg: 1.0, megapixels: 1.0, grounding_px: 768, ref_boost: 1.0 },
};

const MODE_A_LORA = "krea2_identity_edit_v1_2";
const MODE_B_LORAS = ["INPAINTKREA-V1", "Anime to Real", "zoom_krea2edit"];

// Crucial, measured notes. Several of these failure modes are silent.
const NOTES = {
  1: ["CLASSIC EDIT - two reference images", [
    ["It regenerates the WHOLE frame", "Verified: this pipeline samples into a fresh ResolutionSelector-sized latent and has no aspect-preserve restore, so the background changes too. If you need the original background kept pixel-for-pixel, use pipeline 2 instead."],
    ["Reference order matters", "Vision blocks are fed in training order: image = scene/source, reference = subject/donor. Swapping them changes the result."],
    ["Reference method is REQUIRED", "Krea 2 has NO default reference method. This pipeline applies Krea2EditModelPatch. Without one the reference latents are silently IGNORED - measured 0.11 correlation to source vs 0.997 with. Nothing errors; the image just looks unrelated."],
    ["Never stack both methods", "Krea2EditModelPatch OR FluxKontextMultiReferenceLatentMethod. One, never both."],
    ["ref_boost", "Reference-fidelity dial. 1.0 = off; this pipeline is tuned at 4.0."],
    ["grounding_px", "Caps the longest side fed to Qwen3-VL. 512 here; 0 = native."],
    ["Remove background", "RMBG-2.0 on both inputs. Off by default - leave it off for scene-preserving edits like a clothing change."],
  ]],
  2: ["IDENTITY / OSTRIS EDIT - pick MODE, then match the LoRA", [
    ["MODE A - Native Krea2Edit", "Best likeness, and the mode that preserves the untouched area exactly. Krea2EditGroundedEncode + Krea2EditModelPatch, then a native-resolution restore. LoRA must be krea2_identity_edit_v1_2."],
    ["MODE B - Ostris Edit", "TextEncodeKrea2OstrisEdit + FluxKontextMultiReferenceLatentMethod('index_timestep_zero'). That method is REQUIRED - the encoder alone has its reference latents ignored, silently. LoRAs: ai-toolkit only (INPAINTKREA-V1, Anime to Real, zoom_krea2edit)."],
    ["MODE B is not a general editor", "Tested: a maskless clothing instruction under MODE B with INPAINTKREA-V1 returned the source essentially unchanged. It ran clean and errored nothing. Match MODE B to the task its ai-toolkit LoRA was trained for."],
    ["LoRA must match the encoder", "A MODE A LoRA under MODE B (or the reverse) degrades quietly - no error, just a worse image. Mismatches are flagged amber in the LoRA list."],
    ["Clothing changes - use this", "Verified: MODE A, maskless, restore_mode 'smart'. Measured 11.3% of frame changed. Do NOT use a full-body mask in pipeline 3; those fail and leave green limbs."],
    ["restore_mode", "'smart' tries the manual mask, then local auto-detected edits, then the full frame. Everything outside the edit comes back at native resolution."],
  ]],
  3: ["GREEN-MASK INPAINT / OUTPAINT", [
    ["How to paint the mask", "Right-click the SOURCE LoadImage node -> Open in MaskEditor, paint the region, save. Then wire that node's MASK output to this node's mask socket. That is the whole procedure."],
    ["You never paint green yourself", "The MASK is the image's ALPHA channel, not green pixels. ComfyUI's MaskEditor only ever saves an alpha mask - DefaultGreenMaskColor just recolours the swatch it shows you. KreaAspectPreservePrepare paints the actual green region from your alpha mask."],
    ["What the node does with it", "Your mask is grown by 8px, blurred (radius 6 / sigma 4) and fed back as a mask, so the green region has a soft edge. Then prepare paints it green, the Ostris encoder runs with index_timestep_zero, and KreaAspectPreserveRestore puts everything outside the mask back at native resolution."],
    ["OUTPAINT needs no mask", "It pads the canvas, generates its own mask from the padding, and composites the green border itself. The mask socket is ignored and hidden."],
    ["Do not mask whole bodies", "Full-body masks fail here and leave green limbs. For clothing or body changes use pipeline 2 MODE A maskless."],
    ["Removals", "Removals need a Raw checkpoint at cfg ~3 and ~20 steps - the turbo settings here will not remove cleanly."],
  ]],
  4: ["TEXT TO IMAGE", [
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

// Custom-node packs this node calls into, with verified one-click zip links.
// github.com/<repo>/archive/HEAD.zip resolves to the default branch whatever it is
// named, so these survive a master -> main rename. All verified HTTP 200.
// [label, used by, repo url]  — null url = ships with ComfyUI core.
const DEPENDENCIES = [
  ["comfyui-krea2edit", "P1 + P2 MODE A", "https://github.com/lbouaraba/comfyui-krea2edit"],
  ["ComfyUI-Krea2-Ostris-Edit", "P2 MODE B + P3", "https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit"],
  ["ComfyUI-KreaImageAspectPreserve", "P2 restore", "https://github.com/aitrepreneur/ComfyUI-KreaImageAspectPreserve"],
  ["ComfyUI-KreaAspectPreserveOutpaint", "P3 green mask", "https://github.com/aitrepreneur/ComfyUI-KreaAspectPreserveOutpaint"],
  ["rgthree-comfy", "Image Comparer", "https://github.com/rgthree/rgthree-comfy"],
  ["ComfyUI-Impact-Pack", "face detail", "https://github.com/ltdrdata/ComfyUI-Impact-Pack"],
  ["ComfyUI-Impact-Subpack", "face detail", "https://github.com/ltdrdata/ComfyUI-Impact-Subpack"],
  ["ComfyUI-Easy-Use", "P1 rembg", "https://github.com/yolain/ComfyUI-Easy-Use"],
  ["ComfyUI core", "reference method, Klein chain", null],
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
.kaio{--bg:#191919;--panel:#212121;--line:#333;--txt:#dcdcdc;--dim:#8a8a8a;--acc:#4a90d9;
 --acc2:#2d5c8a;--ok:#5a9c5a;--ok2:#3a6b3a;--warn:#e0a33e;
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:12px;color:var(--txt);
 padding:10px;box-sizing:border-box;height:100%;overflow-y:auto;overflow-x:hidden}
.kaio::-webkit-scrollbar{width:8px}.kaio::-webkit-scrollbar-thumb{background:#3a3a3a;border-radius:4px}
.kaio .hd{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.kaio .hd h1{font-size:12px;font-weight:600;letter-spacing:.08em;margin:0;flex:1;
 text-transform:uppercase;color:#fff}
.kaio .tabs{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-bottom:5px}
.kaio .tab{padding:8px 3px;border-radius:7px;background:var(--panel);border:1px solid var(--line);
 cursor:pointer;text-align:center;line-height:1.2;user-select:none;transition:.12s}
.kaio .tab:hover{background:#2a2a2a;border-color:#454545}
.kaio .tab.on{background:var(--acc2);border-color:var(--acc);color:#fff;box-shadow:0 0 0 1px var(--acc) inset}
.kaio .tab b{display:block;font-size:14px}.kaio .tab i{display:block;font-size:9.5px;opacity:.8;font-style:normal}
.kaio .upbar{width:100%;padding:7px;border-radius:7px;background:var(--panel);border:1px solid var(--line);
 color:var(--dim);cursor:pointer;font-size:11px;font-family:inherit;margin-bottom:8px;transition:.12s}
.kaio .upbar:hover{background:#2a2a2a}
.kaio .upbar.on{background:var(--acc2);border-color:var(--acc);color:#fff}
.kaio .sec{margin-bottom:9px}
.kaio label.cap{font-size:9.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);
 display:block;margin-bottom:3px}
.kaio .seg{display:flex;gap:5px}
.kaio .seg button{flex:1;padding:7px 6px;border-radius:6px;background:var(--panel);
 border:1px solid var(--line);color:var(--dim);cursor:pointer;font-size:10.5px;font-family:inherit;line-height:1.3}
.kaio .seg button:hover{background:#2a2a2a}
.kaio .seg button.on{background:var(--ok2);border-color:var(--ok);color:#fff}
.kaio .seg button b{display:block;font-size:11.5px}
.kaio .seg button i{font-style:normal;opacity:.75;font-size:9.5px}
.kaio input[type=text],.kaio input[type=number],.kaio select,.kaio textarea{
 background:#141414;border:1px solid var(--line);border-radius:5px;color:var(--txt);
 padding:6px 7px;font-size:12px;font-family:inherit;box-sizing:border-box;width:100%}
.kaio input:focus,.kaio select:focus,.kaio textarea:focus{outline:none;border-color:var(--acc)}
.kaio textarea{resize:vertical;min-height:74px;line-height:1.4}
.kaio .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.kaio .fld{display:flex;flex-direction:column;min-width:0}
.kaio .hide{display:none!important}
.kaio details.card{background:#1d2530;border:1px solid #2c3d52;border-radius:7px;margin-bottom:9px;overflow:hidden}
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
 margin-bottom:9px;overflow:hidden}
.kaio details.plain>summary{cursor:pointer;padding:7px 10px;font-size:10px;color:var(--dim);
 list-style:none;text-transform:uppercase;letter-spacing:.06em;user-select:none}
.kaio details.plain>summary::-webkit-details-marker{display:none}
.kaio details.plain>summary:hover{background:#282828}
.kaio details.models{background:#232823;border-color:#3a463a}
.kaio details.models>summary{color:#9fc39f;font-weight:600}
.kaio details.models>summary:hover{background:#283028}
.kaio details.plain .body{padding:4px 10px 9px}
.kaio .slot{display:flex;gap:8px;background:var(--panel);border:1px solid var(--line);
 border-radius:7px;padding:7px;margin-bottom:6px}
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
.kaio .chk{display:flex;align-items:center;gap:7px;font-size:11.5px;cursor:pointer}
.kaio .chk input{width:auto;accent-color:var(--acc)}
.kaio .status{font-size:10px;color:var(--dim);border-top:1px solid var(--line);padding-top:7px;
 margin-top:9px;line-height:1.55}
.kaio .status .warn{color:var(--warn);margin-top:3px}
.kaio .savepath{font-family:ui-monospace,Consolas,monospace;font-size:9.5px;color:#8fbf8f;
 margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kaio .savepath.unwired{color:var(--warn)}
.kaio .savepath.unwired::before{content:"! not wired - "}
.kaio .cmdrow{display:flex;gap:6px;align-items:center;margin:3px 0}
.kaio code.cmd{flex:1;background:#141414;border:1px solid var(--line);border-radius:4px;
 padding:5px 7px;font-family:ui-monospace,Consolas,monospace;font-size:9.5px;color:#c8d8c8;
 overflow-x:auto;white-space:nowrap}
.kaio .copybtn{flex:none;padding:5px 9px;border-radius:4px;background:#2a2a2a;
 border:1px solid var(--line);color:var(--dim);cursor:pointer;font-size:9.5px;font-family:inherit}
.kaio .copybtn:hover{background:#333;color:var(--txt)}
.kaio .deprow{display:flex;align-items:baseline;gap:8px;padding:2px 0}
.kaio .deplink{color:#7fb3e8;text-decoration:none;font-size:10.5px;flex:none}
.kaio .deplink:hover{text-decoration:underline;color:#a9cdf5}
.kaio .deplink.builtin{color:var(--dim);cursor:default}
.kaio .depuse{font-size:9.5px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kaio .muted{font-size:9.5px;color:var(--dim);margin:5px 0 3px}
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
    if (!document.getElementById("kaio-css")) {
      const s = el("style"); s.id = "kaio-css"; s.textContent = CSS;
      document.head.appendChild(s);
    }
    // All content lives in an auto-height inner wrapper. The root is the scroll box
    // (its height is driven by the node), so measuring the root tells us nothing —
    // the inner wrapper's height IS the content height.
    this.inner = el("div");
    root.appendChild(this.inner);

    this.build();
    this.sync();

    // Re-fit when a <details> section is toggled, since that changes the height a lot.
    for (const d of [this.note, this.mBox, this.depBox]) {
      d.addEventListener("toggle", () => this.applyHeight());
    }
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
    const fluxRows = (this.fluxWrap && !this.fluxWrap.classList.contains("hide")) ? 4 * 30 + 18 : 0;
    h += (this.mBox.open ? 30 + 3 * 30 + fluxRows : 32) + 9;   // models (top)
    h += (this.note.open ? 30 + this.noteBody.querySelectorAll("dt").length * 46 : 32) + 9;
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
    h += (this.depBox.open ? 30 + (DEPENDENCIES.length + LORAS.length) * 22 + 130 : 32) + 9;
    h += 8 + 16 * (2 + this.status.querySelectorAll(".warn").length);
    return Math.min(Math.max(Math.round(h), 260), 1100);
  }

  applyHeight() {
    const h = this.estimateHeight();
    if (Math.abs(h - (this.node._kaioHeight || 0)) <= 4) return;
    this.node._kaioHeight = h;
    this.node.size[1] = h + 56;
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

  build() {
    const r = this.inner;
    const hd = el("div", "hd");
    hd.appendChild(el("h1", null, "Krea2 AIO AJ"));
    r.appendChild(hd);

    this.tabs = el("div", "tabs"); this.tabEl = {};
    for (const p of PIPES) {
      const t = el("div", "tab");
      t.appendChild(el("b", null, String(p.idx)));
      t.appendChild(el("i", null, p.n));
      t.title = p.label;
      t.onclick = () => { this.setW("pipeline", p.label); this.preset(p.idx); this.sync(); };
      this.tabs.appendChild(t); this.tabEl[p.idx] = t;
    }
    r.appendChild(this.tabs);

    this.upBtn = el("button", "upbar", "5 · FLUX 2 KLEIN UPSCALE");
    this.upBtn.onclick = () => { this.setW("upscale", !this.upscale()); this.sync(); };
    r.appendChild(this.upBtn);

    // Models first — this is where you point it at your checkpoints, so it should be
    // the most obvious thing after picking a pipeline.
    this.mBox = el("details", "plain models");
    this.mBox.open = true;
    this.mBox.appendChild(el("summary", null, "Models"));
    this.mBody = el("div", "body"); this.mBox.appendChild(this.mBody);
    r.appendChild(this.mBox);
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
    this.mBody.appendChild(this.fluxWrap);

    this.note = el("details", "card");
    this.noteSum = el("summary");
    this.noteBody = el("div", "body");
    this.note.appendChild(this.noteSum); this.note.appendChild(this.noteBody);
    r.appendChild(this.note);

    // MODE A/B
    this.modeSec = el("div", "sec");
    this.modeSec.appendChild(el("label", "cap", "Edit mode — encoder + LoRA family"));
    const ms = el("div", "seg");
    this.mA = el("button"); this.mA.appendChild(el("b", null, "MODE A"));
    this.mA.appendChild(el("i", null, "Native Krea2Edit"));
    this.mB = el("button"); this.mB.appendChild(el("b", null, "MODE B"));
    this.mB.appendChild(el("i", null, "Ostris · ai-toolkit"));
    this.mA.onclick = () => { this.setW("edit_mode", "A - Native Krea2Edit (identity)"); this.sync(); };
    this.mB.onclick = () => { this.setW("edit_mode", "B - Ostris Edit (ai-toolkit)"); this.sync(); };
    ms.appendChild(this.mA); ms.appendChild(this.mB);
    this.modeSec.appendChild(ms); r.appendChild(this.modeSec);

    // FILL A/B
    this.fillSec = el("div", "sec");
    this.fillSec.appendChild(el("label", "cap", "Inpaint or outpaint"));
    const fs = el("div", "seg");
    this.fA = el("button"); this.fA.appendChild(el("b", null, "INPAINT"));
    this.fA.appendChild(el("i", null, "you paint the mask"));
    this.fB = el("button"); this.fB.appendChild(el("b", null, "OUTPAINT"));
    this.fB.appendChild(el("i", null, "auto green border"));
    this.fA.onclick = () => { this.setW("fill_mode", "A - INPAINT (you paint the mask)"); this.sync(); };
    this.fB.onclick = () => { this.setW("fill_mode", "B - OUTPAINT (auto green border)"); this.sync(); };
    fs.appendChild(this.fA); fs.appendChild(this.fB);
    this.fillSec.appendChild(fs); r.appendChild(this.fillSec);

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
    r.appendChild(this.padSec);

    // input sockets — images live on real LoadImage nodes outside this one, which is
    // what keeps ComfyUI's MaskEditor working normally
    this.imgSec = el("div", "sec");
    this.imgSec.appendChild(el("label", "cap", "Inputs — wire LoadImage nodes here"));
    this.connList = el("div");
    this.imgSec.appendChild(this.connList);
    this.refWrap = el("label", "chk");
    this.refWrap.style.marginTop = "5px";
    this.useRef = el("input"); this.useRef.type = "checkbox";
    this.useRef.onchange = () => { this.setW("use_reference", this.useRef.checked); this.sync(); };
    this.refWrap.appendChild(this.useRef);
    this.refWrap.appendChild(el("span", null, "Use second reference image"));
    this.imgSec.appendChild(this.refWrap);
    r.appendChild(this.imgSec);

    // prompt
    this.pSec = el("div", "sec");
    this.pSec.appendChild(el("label", "cap", "Prompt / instruction"));
    this.prompt = el("textarea");
    this.prompt.oninput = () => this.setW("prompt", this.prompt.value);
    this.pSec.appendChild(this.prompt); r.appendChild(this.pSec);

    // numbers
    this.numSec = el("div", "sec");
    const g = el("div", "grid"); this.num = {};
    const mk = (name, label, step, min, max) => {
      const f = el("div", "fld");
      f.appendChild(el("label", "cap", label));
      const i = el("input"); i.type = "number";
      if (step != null) i.step = step; if (min != null) i.min = min; if (max != null) i.max = max;
      i.oninput = () => this.setW(name, i.value === "" ? 0 : Number(i.value));
      f.appendChild(i); this.num[name] = { f, i }; g.appendChild(f);
    };
    mk("seed", "Seed", 1, 0); mk("steps", "Steps", 1, 1, 200); mk("cfg", "CFG", 0.1, 0, 30);
    mk("megapixels", "Megapixels", 0.1, 0.1, 16);
    mk("grounding_px", "Grounding px", 64, 0, 4096);
    mk("ref_boost", "Ref boost", 0.01, 0, 1000);
    this.numSec.appendChild(g); r.appendChild(this.numSec);

    // combos
    this.comboSec = el("div", "sec");
    const cg = el("div", "grid"); this.combo = {};
    const mkc = (name, label) => {
      const f = el("div", "fld");
      f.appendChild(el("label", "cap", label));
      const s = el("select");
      s.onchange = () => this.setW(name, s.value);
      f.appendChild(s); this.combo[name] = { f, s }; cg.appendChild(f);
    };
    mkc("sampler", "Sampler"); mkc("scheduler", "Scheduler"); mkc("aspect_ratio", "Aspect");
    this.comboSec.appendChild(cg); r.appendChild(this.comboSec);

    // restore
    this.rSec = el("div", "sec");
    this.rSec.appendChild(el("label", "cap", "Restore mode"));
    this.restore = el("select");
    this.restore.onchange = () => this.setW("restore_mode", this.restore.value);
    this.rSec.appendChild(this.restore); r.appendChild(this.rSec);

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
    r.appendChild(this.lSec);

    // toggles
    this.faceSec = el("div", "sec");
    const fl = el("label", "chk");
    this.face = el("input"); this.face.type = "checkbox";
    this.face.onchange = () => { this.setW("face_detail", this.face.checked); this.sync(); };
    fl.appendChild(this.face); fl.appendChild(el("span", null, "Face detail pass"));
    this.faceSec.appendChild(fl); r.appendChild(this.faceSec);

    this.rbSec = el("div", "sec");
    const rbl = el("label", "chk");
    this.rb = el("input"); this.rb.type = "checkbox";
    this.rb.onchange = () => this.setW("remove_background", this.rb.checked);
    rbl.appendChild(this.rb);
    rbl.appendChild(el("span", null, "Remove background (RMBG-2.0)"));
    this.rbSec.appendChild(rbl); r.appendChild(this.rbSec);



    // output folder — the node derives the subfolder from the active pipeline
    this.saveSec = el("div", "sec");
    this.saveSec.appendChild(el("label", "cap", "Output folder"));
    const sr = el("div", "cmdrow");
    this.saveRoot = el("input"); this.saveRoot.type = "text";
    this.saveRoot.placeholder = "Krea2AJ";
    this.saveRoot.oninput = () => { this.setW("save_root", this.saveRoot.value); this.refreshSavePath(); };
    sr.appendChild(this.saveRoot);
    this.saveSec.appendChild(sr);
    this.savePreview = el("div", "savepath");
    this.saveSec.appendChild(this.savePreview);
    r.appendChild(this.saveSec);

    // required packs + LoRAs, as one-click direct downloads
    this.depBox = el("details", "plain");
    this.depBox.appendChild(el("summary", null, "Required nodes & LoRAs — download links"));
    const depBody = el("div", "body");
    for (const [pack, used, url] of DEPENDENCIES) {
      const row = el("div", "deprow");
      if (url) {
        const a = el("a", "deplink", pack);
        a.href = url + "/archive/HEAD.zip";
        a.target = "_blank"; a.rel = "noopener";
        a.title = "Download " + url + " as a zip";
        row.appendChild(a);
      } else {
        row.appendChild(el("span", "deplink builtin", pack));
      }
      row.appendChild(el("span", "depuse", used));
      depBody.appendChild(row);
    }
    depBody.appendChild(el("div", "muted", "Unzip into ComfyUI/custom_nodes/ and restart."));
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

    // How to get this node onto another machine. The LAN URL only works for people
    // on the same network; for anyone else, send them the zip.
    depBody.appendChild(el("div", "muted", "INSTALL THIS NODE ELSEWHERE"));
    // A loopback host is useless to anyone else, so show a placeholder instead of
    // handing out a URL that only resolves on this machine.
    const LOOPBACK = ["localhost", "127.0.0.1", "::1", ""];
    const host = LOOPBACK.includes(location.hostname) ? "YOUR-LAN-IP" : location.hostname;
    const cmd = `git clone http://${host}:8199/Krea2_AIO_AJ.git KreaUltraController`;
    const cmdRow = el("div", "cmdrow");
    const code = el("code", "cmd", cmd);
    const copy = el("button", "copybtn", "copy");
    copy.onclick = async () => {
      try {
        await navigator.clipboard.writeText(cmd);
        copy.textContent = "copied";
        setTimeout(() => { copy.textContent = "copy"; }, 1200);
      } catch (e) { copy.textContent = "failed"; }
    };
    cmdRow.appendChild(code); cmdRow.appendChild(copy);
    depBody.appendChild(cmdRow);
    depBody.appendChild(el("div", "muted",
      "Run it inside ComfyUI/custom_nodes/, then restart. Same network only — " +
      "for remote friends send them the zip instead."));
    this.depBox.appendChild(depBody);
    r.appendChild(this.depBox);

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
  saveLoras(list) { this.setW("loras_json", JSON.stringify(list)); }

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
      const x = el("button", "lx", "×");
      x.onclick = () => { const l = this.loras(); l.splice(i, 1); this.saveLoras(l); this.renderLoras(); this.updateStatus(); };
      row.appendChild(cb); row.appendChild(nm); row.appendChild(st); row.appendChild(x);
      this.lList.appendChild(row);
    });
  }

  // ---- sync --------------------------------------------------------------
  preset(idx) {
    for (const [k, v] of Object.entries(PRESETS[idx] || {})) {
      this.setW(k, v);
      if (this.num[k]) this.num[k].i.value = v;
    }
  }

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

    // notes
    const key = up ? 5 : p;
    const [title, rows] = NOTES[key] || NOTES[4];
    this.noteSum.textContent = title;
    this.noteBody.innerHTML = "";
    const dl = el("dl");
    for (const [a, b] of rows) { dl.appendChild(el("dt", null, a)); dl.appendChild(el("dd", null, b)); }
    this.noteBody.appendChild(dl);

    this.prompt.value = this.gv("prompt") ?? "";
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
    const outp = String(this.gv("fill_mode") || "").startsWith("B");
    this.fA.classList.toggle("on", !outp);
    this.fB.classList.toggle("on", outp);

    this.renderConnections();
    this.useRef.checked = this.gv("use_reference") !== false;
    this.refWrap.classList.toggle("hide", !shown.has("use_reference"));
    this.face.checked = !!this.gv("face_detail");
    this.rb.checked = !!this.gv("remove_background");

    // Flux loaders appear with the upscale; outpaint padding with pipeline 3 OUTPAINT.
    this.fluxWrap.classList.toggle("hide", !up);
    this.upSteps.value = this.gv("upscale_steps") ?? 2;
    const showPad = (p === 3) && outp;
    this.padSec.classList.toggle("hide", !showPad);
    for (const k of Object.keys(this.pad)) this.pad[k].i.value = this.gv(k) ?? 0;
    if (up && !this.mBox.open) this.mBox.open = true;

    this.saveRoot.value = this.gv("save_root") ?? "Krea2AJ";
    this.refreshSavePath();
    this.renderLoras();
    this.updateStatus();
    this.remeasure();
  }

  remeasure() {
    this.applyHeight();
  }

  refreshSavePath() {
    const p = this.pipe();
    const path = buildSavePath(
      this.gv("save_root"), p,
      String(this.gv("edit_mode") || "").startsWith("B"),
      String(this.gv("fill_mode") || "").startsWith("B"),
      !!this.gv("face_detail"), this.upscale());
    this.savePreview.textContent = "output/" + path + "_00001_.png";
    const wired = (this.node.outputs || []).some(
      (o) => o.name === "save_path" && (o.links || []).length);
    this.savePreview.classList.toggle("unwired", !wired);
    this.savePreview.title = wired ? "save_path is driving your SaveImage node"
      : "Connect the save_path output to SaveImage.filename_prefix for this to take effect";
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

    const label = PIPES.find((x) => x.idx === p)?.label || "?";
    this.status.innerHTML = "";
    this.status.appendChild(el("div", null, `Active: ${label}${up ? "  +  upscale" : ""}`));
    const n = this.loras().filter((x) => x.on).length;
    this.status.appendChild(el("div", null, `${n} LoRA${n === 1 ? "" : "s"} active.`));
    for (const wn of warns) this.status.appendChild(el("div", "warn", "! " + wn));
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

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      if (onNodeCreated) onNodeCreated.apply(this, arguments);
      for (const w of this.widgets || []) hideWidget(w);

      const c = document.createElement("div");
      c.style.boxSizing = "border-box";
      const widget = this.addDOMWidget("kaio_ui", "kaio_ui", c, {
        getValue: () => "", setValue: () => { },
      });
      const self = this;
      // Derive the height from CONTENT, never from self.size. LiteGraph computes the
      // node's size from its widgets, so reading self.size[1] back here creates a
      // feedback loop and the node grows without bound on every redraw.
      widget.computeSize = function (width) {
        const h = self._kaioHeight || 620;
        return [Math.max(10, (width || 520) - 26), h];
      };
      this.size[0] = Math.max(this.size[0] || 0, 480);
      this.size[1] = 700;

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
      setTimeout(() => { try { self._kaio?.sync(); } catch (e) { } }, 48);
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
  },
});
