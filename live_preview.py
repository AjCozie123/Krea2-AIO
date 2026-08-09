"""Krea2 Live Preview (AJ) — a big, resizable window that shows the sampler's live
preview WHILE an image is generating, so you can watch it form instead of waiting.

There is no algorithm here. ComfyUI already streams a live preview over its websocket
during sampling (the small thumbnail on the running node); this node is a display
surface for that same stream, scaled up. All of the work is in js/krea_live_preview.js,
which listens for the frontend's `b_preview` events and paints them large.

The node does not need to execute — the preview stream is global — so it takes no inputs
and produces no outputs. Just drop it on the canvas next to the AIO node and resize it.

NOTE: live previews only appear if ComfyUI's preview method is enabled. Settings ->
"Preview method" = Auto (or TAESD), or launch with --preview-method auto.
"""

from comfy_api.latest import io


class Krea2LivePreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2LivePreview",
            display_name="Krea2 Live Preview (AJ)",
            category="KREA2",
            description=(
                "A large, resizable live view of the sampler preview while generating, "
                "so you can watch the image form. Needs ComfyUI's preview method enabled "
                "(Settings -> Preview method = Auto)."
            ),
            inputs=[],
            outputs=[],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput()
