import logging

from .controller import KreaAIO
from .live_preview import Krea2LivePreview

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# Register the bundled dependency nodes (comfyui-krea2edit, Ostris edit, the two
# aspect-preserve packs) into ComfyUI's global registry, so this pack works from a
# single install. Only fills in node types a separately-installed pack hasn't already
# provided, so there is never a duplicate-registration conflict.
try:
    import nodes as _comfy_nodes
    from . import vendored as _vendored
    _vendored.register(_comfy_nodes.NODE_CLASS_MAPPINGS, _comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _e:  # never let a vendoring hiccup stop the main node from loading
    logging.getLogger(__name__).warning("[KreaAIO] bundled dependency registration skipped: %s", _e)

# HTTP route that auto-fills a LoRA's trigger word from its metadata (for the node UI).
try:
    from . import server_routes as _server_routes
    _server_routes.register_routes()
except Exception as _e:
    logging.getLogger(__name__).warning("[KreaAIO] lora-trigger route not registered: %s", _e)


class KreaAIOExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [KreaAIO, Krea2LivePreview]


async def comfy_entrypoint() -> KreaAIOExtension:
    return KreaAIOExtension()


NODE_CLASS_MAPPINGS = {"KreaAIO": KreaAIO, "Krea2LivePreview": Krea2LivePreview}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KreaAIO": "Krea2 AIO AJ",
    "Krea2LivePreview": "Krea2 Live Preview (AJ)",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
