// Krea2 Prompt Log — embedded UI.
//
// The whole point of this panel is that you can SEE where a prompt will be written before
// you run anything. The destination is resolved by the server (/kaio/prompt_dest), which
// calls the same resolve_folder() the node itself uses, so the preview can never disagree
// with what actually happens on disk.
//
// Like krea_aio.js this does no graph manipulation: it hides the node's real widgets and
// renders a control surface that writes back into them.

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_W = 460;
const NODE_H = 560;

const DESTS = [
  { v: "Auto - output/prompts/<workflow>", n: "AUTO", s: "output/prompts" },
  { v: "Auto + a date subfolder", n: "AUTO + DATE", s: "…/YYYY-MM-DD" },
  { v: "Custom folder (set below)", n: "CUSTOM", s: "you choose" },
];

const WORKFLOWS = [
  { v: "Auto - from the AIO node", n: "Auto", folder: "(from the AIO node)" },
  { v: "1 - Classic edit", n: "1 Classic", folder: "1_Classic" },
  { v: "2 - Identity / Ostris edit", n: "2 Identity", folder: "2_Identity" },
  { v: "3 - Inpaint / Outpaint", n: "3 In/Outpaint", folder: "3_Inpaint_Outpaint" },
  { v: "4 - Text to image", n: "4 Text2Img", folder: "4_TextToImage" },
];

const CSS = `
.kplog{position:relative;
 --bg:#0d1017;--panel:#141a24;--panel2:#1a2130;--field:#0b0e14;--line:#26303f;
 --txt:#e7ecf3;--dim:#8791a1;--acc:#3b82f6;--acc2:#16233a;--acc-b:#4c8dff;--warn:#e0a33e;
 --ok:#4ade80;
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:12px;color:var(--txt);
 background:linear-gradient(180deg,#0f141d 0%,#0b0e14 100%);border-radius:10px;
 padding:11px;box-sizing:border-box;width:100%;height:100%;overflow-y:auto;overflow-x:hidden}
.kplog .hd{display:flex;align-items:center;gap:8px;margin-bottom:9px}
.kplog .hd b{font-size:12.5px;letter-spacing:.02em}
.kplog .hd .sp{flex:1}
/* master switch — the most prominent control on the node */
.kplog .power{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px}
.kplog .power button{padding:9px 6px;border-radius:8px;background:var(--panel2);
 border:1px solid var(--line);color:var(--dim);cursor:pointer;font-family:inherit;
 display:flex;flex-direction:column;align-items:center;gap:2px;line-height:1.2}
.kplog .power button b{font-size:12.5px;font-weight:700;letter-spacing:.04em}
.kplog .power button i{font-style:normal;font-size:9px;opacity:.8}
.kplog .power button:hover{border-color:var(--acc-b)}
.kplog .power button.on{background:#12301e;border-color:var(--ok);color:var(--ok)}
.kplog .power button.off{background:#301616;border-color:#e06a6a;color:#f0a0a0}
.kplog .phint{font-size:9.5px;color:var(--dim);margin-bottom:8px;line-height:1.4}
.kplog .sec{background:var(--panel);border:1px solid var(--line);border-radius:8px;
 padding:8px 9px;margin-bottom:8px}
.kplog .cap{display:block;font-size:9.5px;text-transform:uppercase;letter-spacing:.07em;
 color:var(--dim);margin-bottom:5px}
.kplog .seg{display:grid;grid-auto-flow:column;gap:5px}
.kplog .seg button{padding:6px 4px;border-radius:6px;background:var(--panel2);
 border:1px solid var(--line);color:var(--dim);cursor:pointer;font-family:inherit;
 font-size:10px;line-height:1.25;display:flex;flex-direction:column;align-items:center;gap:1px}
.kplog .seg button b{font-size:10.5px;font-weight:600}
.kplog .seg button i{font-style:normal;font-size:8.5px;opacity:.75}
.kplog .seg button.on{background:var(--acc2);border-color:var(--acc-b);color:#cfe3f7}
.kplog .seg button:hover{border-color:var(--acc-b)}
.kplog input[type=text],.kplog select{width:100%;box-sizing:border-box;background:var(--field);
 border:1px solid var(--line);border-radius:6px;color:var(--txt);padding:5px 7px;
 font-family:inherit;font-size:11px}
.kplog .row{display:flex;align-items:center;gap:8px;margin-top:5px}
.kplog .row span{flex:1;font-size:11px}
.kplog .row input[type=checkbox]{width:auto;accent-color:var(--acc)}
.kplog .grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px}
/* the destination preview — the most important thing on the node */
.kplog .dest{background:var(--field);border:1px solid var(--acc-b);border-radius:8px;
 padding:8px 9px;margin-bottom:8px}
.kplog .dest .lbl{font-size:9px;text-transform:uppercase;letter-spacing:.08em;color:var(--acc-b);
 margin-bottom:4px;display:flex;align-items:center;gap:6px}
.kplog .dest .path{font-family:ui-monospace,Consolas,monospace;font-size:10.5px;
 word-break:break-all;line-height:1.45;color:var(--txt)}
.kplog .dest .file{font-family:ui-monospace,Consolas,monospace;font-size:10.5px;
 color:var(--ok);margin-top:3px}
.kplog .dest .meta{font-size:9.5px;color:var(--dim);margin-top:4px}
.kplog .dest.miss{border-color:var(--warn)}
.kplog .dest.miss .lbl{color:var(--warn)}
.kplog .btn{padding:5px 9px;border-radius:6px;background:var(--panel2);border:1px solid var(--line);
 color:var(--dim);cursor:pointer;font-family:inherit;font-size:10px}
.kplog .btn:hover{background:#25334a;color:#cfe3f7;border-color:var(--acc-b)}
.kplog .last{font-size:9.5px;color:var(--dim);word-break:break-all;line-height:1.4}
.kplog .hide{display:none!important}
`;

function el(t, c, x) {
  const e = document.createElement(t);
  if (c) e.className = c;
  if (x != null) e.textContent = x;
  return e;
}

class PromptLogUI {
  constructor(node, root) {
    this.node = node;
    this.root = root;
    root.className = "kplog";
    if (!document.getElementById("kplog-css")) {
      const s = el("style"); s.id = "kplog-css"; s.textContent = CSS;
      document.head.appendChild(s);
    }
    this.build();
    this.sync();
  }

  w(name) { return this.node.widgets?.find((x) => x.name === name); }
  gv(name) { const w = this.w(name); return w ? w.value : undefined; }
  sv(name, v) { const w = this.w(name); if (w) { w.value = v; w.callback?.(v); } }

  build() {
    const R = this.root;

    const hd = el("div", "hd");
    hd.appendChild(el("b", null, "PROMPT LOG"));
    R.appendChild(hd);

    // ---- the master switch ----
    // Deliberately the biggest control on the node: whether a run gets written at all is
    // the decision you make most often, so it should be readable at a glance and take one
    // click, not a checkbox you have to hunt for.
    this.power = el("div", "power");
    this.onBtn = el("button"); this.onBtn.type = "button";
    this.onBtn.appendChild(el("b", null, "SAVE ON"));
    this.onBtn.appendChild(el("i", null, "log every run"));
    this.onBtn.onclick = () => { this.sv("enabled", true); this.sync(); };
    this.offBtn = el("button"); this.offBtn.type = "button";
    this.offBtn.appendChild(el("b", null, "SAVE OFF"));
    this.offBtn.appendChild(el("i", null, "write nothing"));
    this.offBtn.onclick = () => { this.sv("enabled", false); this.sync(); };
    this.power.appendChild(this.onBtn);
    this.power.appendChild(this.offBtn);
    R.appendChild(this.power);
    this.powerHint = el("div", "phint");
    R.appendChild(this.powerHint);

    // ---- destination preview (server-resolved) ----
    this.dest = el("div", "dest");
    const dl = el("div", "lbl");
    dl.appendChild(el("span", null, "SAVES TO"));
    this.copyBtn = el("button", "btn");
    this.copyBtn.type = "button";
    this.copyBtn.textContent = "copy";
    this.copyBtn.style.marginLeft = "auto";
    this.copyBtn.onclick = async () => {
      try { await navigator.clipboard.writeText(this.folder || ""); this.copyBtn.textContent = "copied"; }
      catch (e) { this.copyBtn.textContent = "select it"; }
      setTimeout(() => { this.copyBtn.textContent = "copy"; }, 1600);
    };
    dl.appendChild(this.copyBtn);
    this.dest.appendChild(dl);
    this.destPath = el("div", "path", "resolving…");
    this.destFile = el("div", "file", "");
    this.destMeta = el("div", "meta", "");
    this.dest.appendChild(this.destPath);
    this.dest.appendChild(this.destFile);
    this.dest.appendChild(this.destMeta);
    R.appendChild(this.dest);

    // ---- where ----
    const s1 = el("div", "sec");
    s1.appendChild(el("label", "cap", "Where"));
    const seg = el("div", "seg");
    this.destBtns = DESTS.map((d) => {
      const b = el("button"); b.type = "button";
      b.appendChild(el("b", null, d.n));
      b.appendChild(el("i", null, d.s));
      b.onclick = () => { this.sv("destination", d.v); this.sync(); };
      seg.appendChild(b);
      return b;
    });
    s1.appendChild(seg);
    this.customWrap = el("div");
    this.customWrap.style.marginTop = "6px";
    this.customWrap.appendChild(el("label", "cap", "Custom folder"));
    this.customIn = el("input"); this.customIn.type = "text";
    this.customIn.placeholder = "D:\\my prompts   (absolute, or relative to ComfyUI)";
    this.customIn.oninput = () => { this.sv("custom_folder", this.customIn.value); this.debouncedResolve(); };
    this.customWrap.appendChild(this.customIn);
    s1.appendChild(this.customWrap);
    R.appendChild(s1);

    // ---- which workflow folder ----
    const s2 = el("div", "sec");
    s2.appendChild(el("label", "cap", "Workflow folder"));
    this.wfSel = el("select");
    WORKFLOWS.forEach((w) => {
      const o = document.createElement("option");
      o.value = w.v; o.textContent = w.n + (w.folder.startsWith("(") ? "  " + w.folder : "  →  " + w.folder);
      this.wfSel.appendChild(o);
    });
    this.wfSel.onchange = () => { this.sv("workflow", this.wfSel.value); this.sync(); };
    s2.appendChild(this.wfSel);
    this.wfHint = el("div", "meta");
    this.wfHint.style.cssText = "font-size:9.5px;color:var(--dim);margin-top:4px;line-height:1.4";
    this.wfHint.textContent =
      "Auto reads the workflow from the AIO node's prompt_json, so each pipeline files itself.";
    s2.appendChild(this.wfHint);
    R.appendChild(s2);

    // ---- how ----
    const s3 = el("div", "sec");
    s3.appendChild(el("label", "cap", "How"));
    const g = el("div", "grid2");
    const mw = el("div");
    mw.appendChild(el("label", "cap", "File mode"));
    this.modeSel = el("select");
    this.modeSel.onchange = () => { this.sv("file_mode", this.modeSel.value); this.sync(); };
    mw.appendChild(this.modeSel);
    const fw = el("div");
    fw.appendChild(el("label", "cap", "Format"));
    this.fmtSel = el("select");
    this.fmtSel.onchange = () => { this.sv("file_format", this.fmtSel.value); this.sync(); };
    fw.appendChild(this.fmtSel);
    g.appendChild(mw); g.appendChild(fw);
    s3.appendChild(g);

    const pf = el("div");
    pf.style.marginTop = "6px";
    pf.appendChild(el("label", "cap", "Filename prefix"));
    this.prefixIn = el("input"); this.prefixIn.type = "text";
    this.prefixIn.oninput = () => { this.sv("filename_prefix", this.prefixIn.value); this.sync(); };
    pf.appendChild(this.prefixIn);
    s3.appendChild(pf);

    const r1 = el("label", "row");
    r1.appendChild(el("span", null, "Record settings (seed, sampler, model, LoRAs)"));
    this.metaChk = el("input"); this.metaChk.type = "checkbox";
    this.metaChk.onchange = () => this.sv("include_metadata", this.metaChk.checked);
    r1.appendChild(this.metaChk);
    s3.appendChild(r1);

    const r2 = el("label", "row");
    r2.appendChild(el("span", null, "Skip if identical to the last one saved"));
    this.dupChk = el("input"); this.dupChk.type = "checkbox";
    this.dupChk.onchange = () => this.sv("skip_duplicates", this.dupChk.checked);
    r2.appendChild(this.dupChk);
    s3.appendChild(r2);
    R.appendChild(s3);

    // ---- last saved ----
    this.lastSec = el("div", "sec");
    this.lastSec.appendChild(el("label", "cap", "Last saved"));
    this.lastTxt = el("div", "last", "nothing yet — queue a run");
    this.lastSec.appendChild(this.lastTxt);
    R.appendChild(this.lastSec);
  }

  fill(sel, widgetName) {
    const w = this.w(widgetName);
    if (!w || !Array.isArray(w.options?.values)) return;
    if (sel.options.length !== w.options.values.length) {
      sel.innerHTML = "";
      w.options.values.forEach((v) => {
        const o = document.createElement("option");
        o.value = v; o.textContent = v;
        sel.appendChild(o);
      });
    }
    sel.value = w.value;
  }

  sync() {
    const on = !!this.gv("enabled");
    this.onBtn.classList.toggle("on", on);
    this.offBtn.classList.toggle("off", !on);
    this.powerHint.textContent = on
      ? "Every generation writes its prompt to the folder below."
      : "Nothing is written. The prompt still passes through to any node wired after this one.";
    // Everything below the switch is irrelevant while saving is off.
    for (const sec of this.root.querySelectorAll(".sec, .dest")) {
      sec.style.opacity = on ? "1" : "0.4";
      sec.style.pointerEvents = on ? "" : "none";
    }
    const dest = String(this.gv("destination") || DESTS[0].v);
    this.destBtns.forEach((b, i) => b.classList.toggle("on", DESTS[i].v === dest));
    this.customWrap.classList.toggle("hide", !dest.startsWith("Custom"));
    this.customIn.value = this.gv("custom_folder") || "";
    this.wfSel.value = this.gv("workflow") || WORKFLOWS[0].v;
    this.fill(this.modeSel, "file_mode");
    this.fill(this.fmtSel, "file_format");
    this.prefixIn.value = this.gv("filename_prefix") || "prompt";
    this.metaChk.checked = !!this.gv("include_metadata");
    this.dupChk.checked = !!this.gv("skip_duplicates");
    this.resolve();
  }

  debouncedResolve() {
    clearTimeout(this._t);
    this._t = setTimeout(() => this.resolve(), 350);
  }

  // Ask the server where this would actually write. Same function the node uses.
  async resolve() {
    const dest = String(this.gv("destination") || "");
    const wf = String(this.gv("workflow") || "");
    const idx = wf.startsWith("Auto") ? 4 : parseInt(wf[0], 10) || 4;
    const q = new URLSearchParams({
      destination: dest,
      custom_folder: this.gv("custom_folder") || "",
      workflow: String(idx),
    });
    try {
      const r = await fetch("/kaio/prompt_dest?" + q.toString());
      const j = await r.json();
      this.folder = j.folder || "";
      this.destPath.textContent = this.folder || "(could not resolve)";
      this.dest.classList.toggle("miss", !j.exists);
      const mode = String(this.gv("file_mode") || "");
      const fmt = String(this.gv("file_format") || "");
      const ext = (fmt.match(/\((\.[a-z]+)\)/) || [])[1] || ".txt";
      const pre = (this.gv("filename_prefix") || "prompt");
      const today = new Date().toISOString().slice(0, 10);
      this.destFile.textContent = mode.startsWith("One file")
        ? `${pre}_0001${ext}, _0002${ext}, …`
        : mode.startsWith("Append to a daily")
          ? `${pre}_${today}${ext}`
          : `${pre}_log${ext}`;
      if (!this.gv("enabled")) {
        this.destFile.textContent = "";
        this.destMeta.textContent = "SAVE is OFF — this run will not be written";
        return;
      }
      this.destMeta.textContent = j.exists
        ? `folder exists · ${j.files} file${j.files === 1 ? "" : "s"} in it`
        : "folder will be created on the first save";
      if (wf.startsWith("Auto")) {
        this.destMeta.textContent +=
          "  ·  Auto: the <workflow> part follows the AIO node";
      }
    } catch (e) {
      this.destPath.textContent = "(could not reach the server)";
    }
  }

  onSaved(d) {
    if (!d) return;
    if (d.skipped === "duplicate") {
      this.lastTxt.textContent = "skipped — identical to the previous prompt";
    } else if (d.skipped === "disabled") {
      this.lastTxt.textContent = "saving is switched off";
    } else if (d.skipped === "empty") {
      this.lastTxt.textContent = "nothing saved — the prompt was empty";
    } else if (d.error) {
      this.lastTxt.textContent = "error: " + d.error;
    } else if (d.saved) {
      this.lastTxt.textContent = d.saved;
    }
    this.resolve();
  }
}

app.registerExtension({
  name: "Krea2PromptLog",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "Krea2PromptLog") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      // Hide the real widgets — the panel drives them.
      for (const w of this.widgets || []) {
        w.computeSize = () => [0, -4];
        w.hidden = true;
        if (w.type && !w.type.startsWith("converted")) w.type = "kplog_hidden";
      }
      const root = document.createElement("div");
      this.addDOMWidget("kplog", "div", root, { serialize: false, hideOnZoom: false });
      this._kplog = new PromptLogUI(this, root);
      this.size = [NODE_W, NODE_H];
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
      try {
        if (output && output.krea2_log) this._kplog?.onSaved(output.krea2_log[0]);
      } catch (e) { console.warn("[Krea2PromptLog]", e); }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      setTimeout(() => { try { this._kplog?.sync(); } catch (e) { } }, 0);
      return r;
    };
  },
});
