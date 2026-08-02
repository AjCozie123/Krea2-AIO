from .controller import KreaAIO

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class KreaAIOExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [KreaAIO]


async def comfy_entrypoint() -> KreaAIOExtension:
    return KreaAIOExtension()


NODE_CLASS_MAPPINGS = {"KreaAIO": KreaAIO}
NODE_DISPLAY_NAME_MAPPINGS = {"KreaAIO": "Krea2 AIO AJ"}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
