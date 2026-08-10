import math
import torch
import torch.nn.functional as F

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None

_BUCKETS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392),
    (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024), (1104, 944), (1184, 880), (1248, 832),
    (1328, 800), (1392, 752), (1456, 720), (1504, 688),
    (1568, 672),
]


def _resize_image(image, height, width, mode="bicubic"):
    x = image.permute(0, 3, 1, 2)
    if mode in ("bicubic", "bilinear"):
        x = F.interpolate(x, size=(int(height), int(width)), mode=mode, align_corners=False)
    else:
        x = F.interpolate(x, size=(int(height), int(width)), mode=mode)
    return x.permute(0, 2, 3, 1)


def _resize_mask(mask, height, width):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return F.interpolate(mask.unsqueeze(1), size=(int(height), int(width)), mode="bilinear", align_corners=False).squeeze(1)


def _prepare_mask(mask, batch, height, width, device, dtype):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError("Expected MASK as [B,H,W] or [H,W]")
    mask = mask.to(device=device, dtype=dtype)
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.repeat(batch, 1, 1)
    if mask.shape[1] != height or mask.shape[2] != width:
        mask = _resize_mask(mask, height, width)
    return mask.clamp(0.0, 1.0)


def _choose_bucket(width, height):
    ratio = width / float(height)
    return min(_BUCKETS, key=lambda p: abs(math.log((p[0] / float(p[1])) / ratio)))


def _blur_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask
    sigma = max(radius / 3.0, 0.5)
    coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    x = mask.unsqueeze(1)
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x.squeeze(1)


def _dilate(mask, amount):
    amount = int(amount)
    if amount <= 0:
        return mask
    k = amount * 2 + 1
    return F.max_pool2d(mask.unsqueeze(1), kernel_size=k, stride=1, padding=amount).squeeze(1)


def _local_color_match(generated, reference, outpaint_mask, strength):
    strength = float(strength)
    if strength <= 0.0:
        return generated
    ring_width = max(24, int(round(min(generated.shape[1], generated.shape[2]) * 0.035)))
    outer = _dilate(outpaint_mask, ring_width)
    ring = (outer - outpaint_mask).clamp(0.0, 1.0)
    weight = ring.unsqueeze(-1)
    count = weight.sum(dim=(1, 2), keepdim=True)
    valid = (count >= 64.0).to(generated.dtype)
    denom = count.clamp_min(64.0)
    mean_g = (generated * weight).sum(dim=(1, 2), keepdim=True) / denom
    mean_r = (reference * weight).sum(dim=(1, 2), keepdim=True) / denom
    var_g = (((generated - mean_g) ** 2) * weight).sum(dim=(1, 2), keepdim=True) / denom
    var_r = (((reference - mean_r) ** 2) * weight).sum(dim=(1, 2), keepdim=True) / denom
    std_g = var_g.clamp_min(1e-6).sqrt()
    std_r = var_r.clamp_min(1e-6).sqrt()
    matched = (generated - mean_g) * (std_r / std_g).clamp(0.75, 1.35) + mean_r
    mixed = generated * (1.0 - strength) + matched * strength
    return generated * (1.0 - valid) + mixed * valid


def _get_work_device(default_device):
    if model_management is None:
        return default_device
    try:
        dev = model_management.get_torch_device()
        return dev if dev is not None else default_device
    except Exception:
        return default_device


class KreaAspectPreservePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "green_outpaint_canvas": ("IMAGE",),
                "outpaint_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "KREA_ASPECT_MAP", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("prepared_image", "prepared_mask", "restore_map", "width", "height", "batch_size", "prepare_info")
    FUNCTION = "prepare"
    CATEGORY = "Krea 2/Outpaint"

    def prepare(self, green_outpaint_canvas, outpaint_mask):
        if green_outpaint_canvas.ndim != 4:
            raise ValueError("Expected IMAGE as [B,H,W,C]")
        batch, src_h, src_w, channels = green_outpaint_canvas.shape
        if batch != 1:
            raise ValueError("This node currently expects one input image")

        mask = _prepare_mask(
            outpaint_mask, batch, src_h, src_w,
            green_outpaint_canvas.device, green_outpaint_canvas.dtype,
        )

        bucket_w, bucket_h = _choose_bucket(src_w, src_h)
        scale = min(bucket_w / float(src_w), bucket_h / float(src_h))
        fitted_w = max(1, min(bucket_w, int(round(src_w * scale))))
        fitted_h = max(1, min(bucket_h, int(round(src_h * scale))))

        pad_left = (bucket_w - fitted_w) // 2
        pad_right = bucket_w - fitted_w - pad_left
        pad_top = (bucket_h - fitted_h) // 2
        pad_bottom = bucket_h - fitted_h - pad_top

        fitted_image = _resize_image(green_outpaint_canvas, fitted_h, fitted_w, "bicubic")
        fitted_mask = _resize_mask(mask, fitted_h, fitted_w).clamp(0.0, 1.0)

        image_nchw = fitted_image.permute(0, 3, 1, 2)
        image_nchw = F.pad(image_nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        prepared = image_nchw.permute(0, 2, 3, 1)

        mask_nchw = fitted_mask.unsqueeze(1)
        mask_nchw = F.pad(mask_nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        prepared_mask = mask_nchw.squeeze(1).clamp(0.0, 1.0)

        hard = (prepared_mask > 0.5).unsqueeze(-1)
        pure_green = torch.zeros_like(prepared)
        pure_green[..., 1] = 1.0
        prepared = torch.where(hard, pure_green, prepared)

        restore_map = {
            "source_width": int(src_w),
            "source_height": int(src_h),
            "bucket_width": int(bucket_w),
            "bucket_height": int(bucket_h),
            "fitted_width": int(fitted_w),
            "fitted_height": int(fitted_h),
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
        }
        info = (
            f"native={src_w}x{src_h}; bucket={bucket_w}x{bucket_h}; "
            f"fitted={fitted_w}x{fitted_h}; padding="
            f"L{pad_left}/R{pad_right}/T{pad_top}/B{pad_bottom}; "
            f"uniform_scale={scale:.6f}"
        )
        return prepared, prepared_mask, restore_map, int(bucket_w), int(bucket_h), int(batch), info


class KreaAspectPreserveRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expanded_original": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "outpaint_mask": ("MASK",),
                "restore_map": ("KREA_ASPECT_MAP",),
                "seam_overlap": ("INT", {"default": 6, "min": 0, "max": 128, "step": 1}),
                "seam_feather": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "color_match_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("final_image", "restored_generated", "restore_info")
    FUNCTION = "restore"
    CATEGORY = "Krea 2/Outpaint"

    def restore(self, expanded_original, generated_image, outpaint_mask, restore_map,
                seam_overlap, seam_feather, color_match_strength):
        if expanded_original.ndim != 4 or generated_image.ndim != 4:
            raise ValueError("Expected IMAGE tensors as [B,H,W,C]")
        batch, target_h, target_w, channels = expanded_original.shape
        if batch != 1:
            raise ValueError("This node currently expects one input image")
        if generated_image.shape[0] != 1:
            raise ValueError("Expected one generated image")
        if target_w != restore_map["source_width"] or target_h != restore_map["source_height"]:
            raise ValueError("expanded_original dimensions do not match the preparation map")

        out_device = expanded_original.device
        out_dtype = expanded_original.dtype
        work_device = _get_work_device(out_device)
        work_dtype = torch.float32

        print(f"[Krea Aspect Restore] Starting on {work_device}: {target_w}x{target_h}")
        # Use blocking transfers. The previous GPU-fast build used asynchronous
        # non_blocking copies and could return partially copied tensors, producing
        # intermittent multicolored horizontal stripes with no Python error.
        expanded = expanded_original.to(device=work_device, dtype=work_dtype)
        gen = generated_image.to(device=work_device, dtype=work_dtype)

        gen_h, gen_w = gen.shape[1], gen.shape[2]
        bucket_w = restore_map["bucket_width"]
        bucket_h = restore_map["bucket_height"]
        fitted_w = restore_map["fitted_width"]
        fitted_h = restore_map["fitted_height"]
        pad_left = restore_map["pad_left"]
        pad_top = restore_map["pad_top"]

        ys = torch.arange(target_h, device=work_device, dtype=work_dtype)
        xs = torch.arange(target_w, device=work_device, dtype=work_dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        if target_w > 1:
            prepared_x = pad_left + xx * ((fitted_w - 1) / float(target_w - 1))
        else:
            prepared_x = torch.full_like(xx, float(pad_left))
        if target_h > 1:
            prepared_y = pad_top + yy * ((fitted_h - 1) / float(target_h - 1))
        else:
            prepared_y = torch.full_like(yy, float(pad_top))

        if gen_w > 1:
            grid_x = (prepared_x / float(bucket_w - 1)) * 2.0 - 1.0
        else:
            grid_x = torch.zeros_like(prepared_x)
        if gen_h > 1:
            grid_y = (prepared_y / float(bucket_h - 1)) * 2.0 - 1.0
        else:
            grid_y = torch.zeros_like(prepared_y)

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        restored = F.grid_sample(
            gen.permute(0, 3, 1, 2),
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        ).permute(0, 2, 3, 1).clamp(0.0, 1.0)

        mask = _prepare_mask(outpaint_mask, batch, target_h, target_w, work_device, work_dtype)
        restored = _local_color_match(restored, expanded, mask, color_match_strength).clamp(0.0, 1.0)

        alpha = _dilate(mask, int(seam_overlap))
        alpha = _blur_mask(alpha, int(seam_feather)).clamp(0.0, 1.0).unsqueeze(-1)
        final = restored * alpha + expanded * (1.0 - alpha)

        # Sanitize before transfer and force completion of all GPU work.
        final = torch.nan_to_num(final, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        restored = torch.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if work_device.type == "cuda":
            torch.cuda.synchronize(work_device)

        # Blocking device copies are intentional. Returning an asynchronous GPU->CPU
        # copy lets downstream preview/save nodes read incomplete image memory.
        final = final.to(device=out_device, dtype=out_dtype).contiguous()
        restored = restored.to(device=out_device, dtype=out_dtype).contiguous()
        if out_device.type == "cuda":
            torch.cuda.synchronize(out_device)
        print(f"[Krea Aspect Restore] Finished on {work_device}")
        info = (
            f"restored native={target_w}x{target_h} from generated={gen_w}x{gen_h}; "
            f"bucket={bucket_w}x{bucket_h}; fitted={fitted_w}x{fitted_h}; "
            f"seam_overlap={seam_overlap}; seam_feather={seam_feather}; "
            f"color_match={color_match_strength:.2f}; work_device={str(work_device)}"
        )
        return final.clamp(0.0, 1.0), restored, info


NODE_CLASS_MAPPINGS = {
    "KreaAspectPreservePrepare": KreaAspectPreservePrepare,
    "KreaAspectPreserveRestore": KreaAspectPreserveRestore,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KreaAspectPreservePrepare": "Krea Aspect-Preserve Prepare",
    "KreaAspectPreserveRestore": "Krea Aspect-Preserve Restore",
}
