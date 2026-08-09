// Krea2 Green Mask Enforcer
// -------------------------------------------------------------------------------------
// Forces ComfyUI's Mask Editor paint colour to neon green (#00FF00) reliably, on any
// frontend version, for EVERY user who installs this pack. Painting a mask is far easier
// to see in bright green, and this pipeline's inpaint/outpaint is built around a green
// region, so the editor should always open green.
//
// Why a dedicated enforcer (instead of setting the colour once):
//   1. The colour <input> can mount a beat AFTER the editor root appears — a single
//      early set finds no input and does nothing.
//   2. The editor reads its colour at init; if green is set a moment too late, the editor
//      keeps its default and never re-reads.
//   3. The editor de-dupes by value — re-setting the same value is ignored, so a plain
//      "set to green" can be a no-op when the input already shows green but the canvas
//      is still on the default colour.
//
// This enforcer beats all three: it watches for the editor, RETRIES until the input
// exists, FORCES adoption with a one-tick jiggle (near-green -> exact green) so the
// change handler always fires, and keeps a light guard that puts green back if anything
// resets it while the editor is open. Setting the value through the native setter and
// dispatching input + change is exactly what a real user picking the colour does.
//
// It is global (loads with the pack), so you do NOT need a separate green-mask node.

const { app } = window.comfyAPI.app;

const GREEN = "#00ff00";
const NEAR_GREEN = "#01ff00"; // visually identical; used only to force a change event
const ROOT_SELECTORS = [
  '[data-testid="mask-editor-root"]',
  ".maskEditor-dialog-root",
  "#maskEditor",
].join(",");

function nativeSet(input, value) {
  const d = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
  if (d && d.set) d.set.call(input, value);
  else input.value = value;
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function forceGreen(input) {
  // Jiggle defeats value de-duplication and forces the editor's change handler to run,
  // ending on exact #00FF00.
  nativeSet(input, NEAR_GREEN);
  nativeSet(input, GREEN);
}

function enforceOn(root) {
  if (!root || root.__kreaGreenGuard) return;

  let tries = 0;
  const maxTries = 40; // ~4s at 100ms — covers slow first-open mounts

  const tick = () => {
    if (!document.documentElement.contains(root)) return; // editor was closed
    const input = root.querySelector('input[type="color"]');

    if (input && !root.__kreaGreenGuard) {
      root.__kreaGreenGuard = true;
      forceGreen(input);

      // Light guard: if the colour ever drifts off green while the editor is open, put
      // it back. Cleans itself up when the editor closes.
      const guard = setInterval(() => {
        if (!document.documentElement.contains(root)) {
          clearInterval(guard);
          return;
        }
        const inp = root.querySelector('input[type="color"]');
        if (inp && (inp.value || "").toLowerCase() !== GREEN) forceGreen(inp);
      }, 400);
      return; // done; guard takes over
    }

    tries += 1;
    if (tries < maxTries) setTimeout(tick, 100);
    else console.warn("[Krea2 Green Mask] Mask Editor opened but no colour input was found.");
  };

  tick();
}

function scan(node) {
  node = node || document;
  if (node instanceof Element && node.matches && node.matches(ROOT_SELECTORS)) enforceOn(node);
  if (node.querySelectorAll) node.querySelectorAll(ROOT_SELECTORS).forEach(enforceOn);
}

let observer = null;
function start() {
  scan(document);
  observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const n of m.addedNodes) if (n instanceof Element) scan(n);
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

app.registerExtension({
  name: "Krea2.GreenMaskEnforcer",
  async setup() {
    if (!document.body) {
      window.addEventListener("DOMContentLoaded", () => start(), { once: true });
    } else {
      start();
    }
    console.info("[Krea2 Green Mask] Enforcer active — Mask Editor paint colour pinned to #00FF00.");
  },
});
