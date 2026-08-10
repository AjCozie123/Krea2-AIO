"""Call ComfyUI node implementations directly from Python.

The pipelines in this package are NOT reimplementations. Every step looks the real
node class up in ComfyUI's live registry (nodes.NODE_CLASS_MAPPINGS) and calls it, so
Krea2EditModelPatch, TextEncodeKrea2OstrisEdit, KSampler and friends run exactly the
same code the graph runs. If a custom node is updated, this follows it automatically.

Two node styles have to be handled:
  V1  class has INPUT_TYPES / RETURN_TYPES / FUNCTION = "name"; needs an instance.
  V3  class subclasses comfy_api.latest.io.ComfyNode; execute() is a classmethod and
      returns io.NodeOutput.
"""

import logging

log = logging.getLogger(__name__)


class NodeCallError(RuntimeError):
    pass


def _mappings():
    import nodes
    return nodes.NODE_CLASS_MAPPINGS


def get_class(node_type):
    cls = _mappings().get(node_type)
    if cls is None:
        raise NodeCallError(
            f"node type {node_type!r} is not installed / failed to import. "
            f"The AIO node needs it; check the ComfyUI console for that pack's import error."
        )
    return cls


def has(node_type):
    """True if a node type is registered. Used to make OPTIONAL features (face detail,
    background removal, upscale colour-match) degrade gracefully instead of erroring when
    their pack isn't installed."""
    return _mappings().get(node_type) is not None


def _is_v3(cls):
    # V3 nodes expose define_schema(); V1 nodes expose INPUT_TYPES().
    return hasattr(cls, "define_schema") and hasattr(cls, "execute")


def call(node_type, **kwargs):
    """Run a node and return its outputs as a tuple (always a tuple)."""
    cls = get_class(node_type)

    if _is_v3(cls):
        # V3 nodes read class-level context the executor installs before execute() —
        # e.g. GetImageSize does `if cls.hidden.unique_id`. Calling cls.execute()
        # directly leaves cls.hidden as None and blows up with an AttributeError that
        # looks nothing like the real cause. PREPARE_CLASS_CLONE(None) builds the same
        # clone the executor uses, with every hidden value set to None.
        target = cls
        try:
            if hasattr(cls, "PREPARE_CLASS_CLONE"):
                target = cls.PREPARE_CLASS_CLONE(None)
        except Exception as e:
            log.debug("[KreaAIO] PREPARE_CLASS_CLONE failed for %s: %s", node_type, e)
        try:
            result = target.execute(**kwargs)
        except TypeError as e:
            raise NodeCallError(f"{node_type}.execute() rejected {sorted(kwargs)}: {e}") from e
        # io.NodeOutput exposes the positional results on .result
        out = getattr(result, "result", None)
        if out is None:
            out = result
        return out if isinstance(out, tuple) else (out,)

    fn_name = getattr(cls, "FUNCTION", None)
    if not fn_name:
        raise NodeCallError(f"{node_type} has no FUNCTION attribute and is not a V3 node")
    inst = cls()
    fn = getattr(inst, fn_name, None)
    if fn is None:
        raise NodeCallError(f"{node_type}.{fn_name} does not exist")
    try:
        out = fn(**kwargs)
    except TypeError as e:
        raise NodeCallError(f"{node_type}.{fn_name}() rejected {sorted(kwargs)}: {e}") from e

    # V1 nodes may return a dict for UI-producing nodes
    if isinstance(out, dict):
        out = out.get("result", ())
    return out if isinstance(out, tuple) else (out,)


def call1(node_type, **kwargs):
    """Run a node and return only its first output."""
    return call(node_type, **kwargs)[0]


def preview_images(images):
    """Show the result on the node itself, without saving it.

    Saving belongs to a SaveImage node wired to the image output. PreviewImage returns
    {"ui": {...}} rather than a result tuple, so it cannot go through call().
    """
    try:
        cls = get_class("PreviewImage")
    except NodeCallError:
        return {}
    inst = cls()
    fn = getattr(inst, getattr(cls, "FUNCTION", "save_images"))
    out = fn(images=images)
    if isinstance(out, dict):
        return out.get("ui", {})
    return {}


def supports(node_type, input_name):
    """True if a node declares the given input — lets pipelines adapt to pack versions."""
    try:
        cls = get_class(node_type)
    except NodeCallError:
        return False
    try:
        if _is_v3(cls):
            return any(i.id == input_name for i in cls.define_schema().inputs)
        spec = cls.INPUT_TYPES()
        return input_name in (spec.get("required") or {}) or input_name in (spec.get("optional") or {})
    except Exception:
        return False


def only_supported(node_type, kwargs):
    """Drop any kwargs the INSTALLED node version doesn't accept.

    A user may have an OLDER copy of a dependency pack installed (which takes precedence
    over the bundled one), missing newer optional inputs like Krea2EditModelPatch's
    'target_latent'. Passing those raises 'unexpected keyword argument'. Filtering keeps
    the node working across versions; the dropped inputs are optional refinements. If the
    node type can't be inspected, pass everything through unchanged.
    """
    try:
        return {k: v for k, v in kwargs.items() if supports(node_type, k)}
    except Exception:
        return kwargs
