# Bundled dependencies — credits

Krea2 AIO AJ bundles verbatim copies of the following node packs so it works from a
single install. All are redistributed under their original licenses (full texts in
`LICENSES/`). All credit for these implementations goes to their authors.

| Bundled module | Original pack | Author | License |
|---|---|---|---|
| `krea2edit.py` | [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) | lbouaraba | Apache-2.0 |
| `ostris_edit.py` | [ComfyUI-Krea2-Ostris-Edit](https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit) | ostris | MIT |
| `krea_aspect.py` | [ComfyUI-KreaImageAspectPreserve](https://github.com/aitrepreneur/ComfyUI-KreaImageAspectPreserve) | aitrepreneur | Apache-2.0 |
| `krea_outpaint.py` | [ComfyUI-KreaAspectPreserveOutpaint](https://github.com/aitrepreneur/ComfyUI-KreaAspectPreserveOutpaint) | aitrepreneur | Apache-2.0 |

If you already have any of these installed as separate custom nodes, the installed copy
takes precedence — the bundled copy is only used as a fallback so nothing conflicts.

Optional features still require their own packs (they are large and/or have extra pip
dependencies, so they are not bundled):
- **Face detail** → ComfyUI-Impact-Pack + ComfyUI-Impact-Subpack
- **Background removal (pipeline 1)** → ComfyUI-Easy-Use
- **Color match on the Flux upscale** → ComfyUI-KJNodes
- **Before/after comparer in the example workflow** → rgthree-comfy

Without those, the node still runs — it just skips that optional step.
