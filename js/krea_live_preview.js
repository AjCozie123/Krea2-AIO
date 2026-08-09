// Krea2 Live Preview (AJ) — a big window that shows the sampler's live preview while an
// image is generating. ComfyUI streams preview frames over its websocket during
// sampling and the frontend re-emits them as `b_preview` events (a Blob per frame). We
// simply paint the latest frame, scaled to fill this node, plus a thin progress bar.
//
// This node never runs in the graph — the preview stream is global — so it just needs to
// exist on the canvas to host the display. Live previews require ComfyUI's preview
// method to be enabled (Settings -> Preview method = Auto, or --preview-method auto).

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

app.registerExtension({
  name: "Krea2LivePreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Krea2LivePreview") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      if (onNodeCreated) onNodeCreated.apply(this, arguments);

      const wrap = document.createElement("div");
      wrap.style.cssText =
        "width:100%;height:100%;display:flex;align-items:center;justify-content:center;" +
        "background:#111;border-radius:6px;overflow:hidden;position:relative;box-sizing:border-box";

      const img = document.createElement("img");
      img.style.cssText =
        "max-width:100%;max-height:100%;object-fit:contain;image-rendering:auto;display:none";

      const hint = document.createElement("div");
      hint.textContent = "Live sampler preview — the image appears here while generating.";
      hint.style.cssText =
        "color:#666;font:12px system-ui,-apple-system,'Segoe UI',sans-serif;text-align:center;" +
        "padding:12px;line-height:1.5;max-width:80%";

      const bar = document.createElement("div");
      bar.style.cssText =
        "position:absolute;left:0;bottom:0;height:3px;background:#4a90d9;width:0%;transition:width .12s";

      const pct = document.createElement("div");
      pct.style.cssText =
        "position:absolute;right:6px;bottom:6px;color:#cfe3f7;font:11px ui-monospace,Consolas,monospace;" +
        "background:rgba(0,0,0,.45);padding:1px 6px;border-radius:4px;display:none";

      wrap.appendChild(img);
      wrap.appendChild(hint);
      wrap.appendChild(bar);
      wrap.appendChild(pct);

      this.addDOMWidget("krea_live", "krea_live", wrap, {
        getValue: () => "",
        setValue: () => {},
      });

      // Start large; the user can drag it bigger.
      this.size = [520, 580];
      this.resizable = true;

      let url = null;

      this._kreaPreview = (e) => {
        const blob = e.detail;
        if (!(blob instanceof Blob)) return;
        if (url) URL.revokeObjectURL(url);
        url = URL.createObjectURL(blob);
        img.src = url;
        img.style.display = "";
        hint.style.display = "none";
      };

      this._kreaProgress = (e) => {
        const d = e.detail || {};
        const max = Number(d.max) || 0;
        const val = Number(d.value) || 0;
        const p = max ? Math.min(100, Math.round((val / max) * 100)) : 0;
        bar.style.width = p + "%";
        pct.textContent = p + "%";
        pct.style.display = p > 0 && p < 100 ? "" : "none";
        if (p >= 100) setTimeout(() => { bar.style.width = "0%"; pct.style.display = "none"; }, 500);
      };

      api.addEventListener("b_preview", this._kreaPreview);
      api.addEventListener("progress", this._kreaProgress);
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      try {
        if (this._kreaPreview) api.removeEventListener("b_preview", this._kreaPreview);
        if (this._kreaProgress) api.removeEventListener("progress", this._kreaProgress);
      } catch (e) { /* ignore */ }
      if (onRemoved) onRemoved.apply(this, arguments);
    };
  },
});
