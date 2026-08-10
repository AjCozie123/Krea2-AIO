"""Bundled dependency nodes, so Krea2 AIO AJ works from a single install.

These modules are VERBATIM copies of four small, permissively-licensed node packs the
AIO pipelines call into. Bundling them means a user only has to install this one node —
they no longer need to separately git-clone the packs below. Full license texts are in
vendored/LICENSES/ and credit is in vendored/NOTICE.md.

    krea2edit.py     comfyui-krea2edit               (Apache-2.0)  Krea2EditModelPatch,
                                                                   Krea2EditGroundedEncode
    ostris_edit.py   ComfyUI-Krea2-Ostris-Edit       (MIT)         TextEncodeKrea2OstrisEdit (+patch)
    krea_aspect.py   ComfyUI-KreaImageAspectPreserve  (Apache-2.0)  KreaImageAspectPreservePrepare,
                                                                   KreaImageUniversalAspectPreserveRestore
    krea_outpaint.py ComfyUI-KreaAspectPreserveOutpaint (Apache-2.0) KreaAspectPreservePrepare,
                                                                   KreaAspectPreserveRestore

register() adds each class to ComfyUI's global registry ONLY if that node type is not
already present — so if a user also has the real pack installed, THEIRS wins and there is
no duplicate-registration conflict. Each module is imported in its own try/except, so one
that fails to import (e.g. an older ComfyUI missing a comfy internal) never breaks the
rest of the node.
"""

import importlib
import logging

log = logging.getLogger(__name__)

# module name -> the pack it came from (for clear logging)
_MODULES = {
    "krea2edit": "comfyui-krea2edit",
    "ostris_edit": "ComfyUI-Krea2-Ostris-Edit",
    "krea_aspect": "ComfyUI-KreaImageAspectPreserve",
    "krea_outpaint": "ComfyUI-KreaAspectPreserveOutpaint",
}


def register(class_mappings, display_mappings=None):
    """Merge bundled node classes into ComfyUI's registry without clobbering real packs."""
    added = []
    for modname, pack in _MODULES.items():
        try:
            mod = importlib.import_module("." + modname, __name__)
        except Exception as e:
            log.warning("[KreaAIO] bundled %s failed to load (%s). Its pipeline(s) will only "
                        "work if you install %s separately.", modname, e, pack)
            continue
        ncm = getattr(mod, "NODE_CLASS_MAPPINGS", {}) or {}
        ndm = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
        for key, cls in ncm.items():
            if key in class_mappings:
                continue  # a separately-installed pack already provides it — leave it alone
            class_mappings[key] = cls
            if display_mappings is not None and key in ndm:
                display_mappings.setdefault(key, ndm[key])
            added.append(key)
    if added:
        log.info("[KreaAIO] bundled dependency nodes registered: %s", ", ".join(sorted(added)))
    return added
