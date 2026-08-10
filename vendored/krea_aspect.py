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


def _choose_bucket(width, height):
    ratio = width / float(height)
    return min(_BUCKETS, key=lambda p: abs(math.log((p[0] / float(p[1])) / ratio)))


def _resize_image(image, height, width, mode="bicubic"):
    x = image.permute(0, 3, 1, 2)
    if mode in ("bicubic", "bilinear"):
        x = F.interpolate(x, size=(int(height), int(width)), mode=mode, align_corners=False)
    else:
        x = F.interpolate(x, size=(int(height), int(width)), mode=mode)
    return x.permute(0, 2, 3, 1)


def _get_work_device(default_device):
    if model_management is None:
        return default_device
    try:
        device = model_management.get_torch_device()
        return device if device is not None else default_device
    except Exception:
        return default_device


class KreaImageAspectPreservePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "KREA_IMAGE_ASPECT_MAP", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("prepared_image", "restore_map", "width", "height", "batch_size", "prepare_info")
    FUNCTION = "prepare"
    CATEGORY = "Krea 2/Native Resolution"

    def prepare(self, image):
        if image.ndim != 4:
            raise ValueError("Expected IMAGE as [B,H,W,C]")
        batch, src_h, src_w, channels = image.shape
        if batch != 1:
            raise ValueError("This node currently expects one input image")

        bucket_w, bucket_h = _choose_bucket(src_w, src_h)
        scale = min(bucket_w / float(src_w), bucket_h / float(src_h))
        fitted_w = max(1, min(bucket_w, int(round(src_w * scale))))
        fitted_h = max(1, min(bucket_h, int(round(src_h * scale))))

        pad_left = (bucket_w - fitted_w) // 2
        pad_right = bucket_w - fitted_w - pad_left
        pad_top = (bucket_h - fitted_h) // 2
        pad_bottom = bucket_h - fitted_h - pad_top

        fitted = _resize_image(image, fitted_h, fitted_w, "bicubic")
        prepared = F.pad(
            fitted.permute(0, 3, 1, 2),
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="replicate",
        ).permute(0, 2, 3, 1).contiguous()

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
        return prepared, restore_map, int(bucket_w), int(bucket_h), int(batch), info


class KreaImageAspectPreserveRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "restore_map": ("KREA_IMAGE_ASPECT_MAP",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("final_image", "restore_info")
    FUNCTION = "restore"
    CATEGORY = "Krea 2/Native Resolution"

    def restore(self, original_image, generated_image, restore_map):
        if original_image.ndim != 4 or generated_image.ndim != 4:
            raise ValueError("Expected IMAGE tensors as [B,H,W,C]")
        batch, target_h, target_w, channels = original_image.shape
        if batch != 1 or generated_image.shape[0] != 1:
            raise ValueError("This node currently expects one image")
        if target_w != restore_map["source_width"] or target_h != restore_map["source_height"]:
            raise ValueError("original_image dimensions do not match restore_map")

        out_device = original_image.device
        out_dtype = original_image.dtype
        work_device = _get_work_device(out_device)
        work_dtype = torch.float32

        print(f"[Krea Image Restore] Starting on {work_device}: {target_w}x{target_h}")
        gen = generated_image.to(device=work_device, dtype=work_dtype)
        bucket_w = int(restore_map["bucket_width"])
        bucket_h = int(restore_map["bucket_height"])
        fitted_w = int(restore_map["fitted_width"])
        fitted_h = int(restore_map["fitted_height"])
        pad_left = int(restore_map["pad_left"])
        pad_top = int(restore_map["pad_top"])

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

        grid_x = (prepared_x / float(max(bucket_w - 1, 1))) * 2.0 - 1.0
        grid_y = (prepared_y / float(max(bucket_h - 1, 1))) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        restored = F.grid_sample(
            gen.permute(0, 3, 1, 2),
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        ).permute(0, 2, 3, 1)

        restored = torch.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if work_device.type == "cuda":
            torch.cuda.synchronize(work_device)

        restored = restored.to(device=out_device, dtype=out_dtype).contiguous()
        if out_device.type == "cuda":
            torch.cuda.synchronize(out_device)
        print(f"[Krea Image Restore] Finished on {work_device}")

        info = (
            f"restored native={target_w}x{target_h} from generated="
            f"{generated_image.shape[2]}x{generated_image.shape[1]}; "
            f"bucket={bucket_w}x{bucket_h}; fitted={fitted_w}x{fitted_h}; "
            f"work_device={str(work_device)}"
        )
        return restored, info





def _prepare_mask(mask, batch, height, width, device, dtype):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError("Expected MASK as [B,H,W] or [H,W]")
    mask = mask.to(device=device, dtype=dtype)
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.repeat(batch, 1, 1)
    if mask.shape[1] != height or mask.shape[2] != width:
        mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
    return mask.clamp(0.0, 1.0)


def _dilate(mask, amount):
    amount = int(amount)
    if amount <= 0:
        return mask
    k = amount * 2 + 1
    return F.max_pool2d(mask.unsqueeze(1), kernel_size=k, stride=1, padding=amount).squeeze(1)


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


def _local_color_match(generated, reference, mask, strength):
    strength = float(strength)
    if strength <= 0.0:
        return generated
    ring_width = max(12, int(round(min(generated.shape[1], generated.shape[2]) * 0.02)))
    outer = _dilate(mask, ring_width)
    ring = (outer - mask).clamp(0.0, 1.0)
    weight = ring.unsqueeze(-1)
    count = weight.sum(dim=(1, 2), keepdim=True)
    valid = (count >= 32.0).to(generated.dtype)
    denom = count.clamp_min(32.0)
    mean_g = (generated * weight).sum(dim=(1, 2), keepdim=True) / denom
    mean_r = (reference * weight).sum(dim=(1, 2), keepdim=True) / denom
    var_g = (((generated - mean_g) ** 2) * weight).sum(dim=(1, 2), keepdim=True) / denom
    var_r = (((reference - mean_r) ** 2) * weight).sum(dim=(1, 2), keepdim=True) / denom
    std_g = var_g.clamp_min(1e-6).sqrt()
    std_r = var_r.clamp_min(1e-6).sqrt()
    matched = (generated - mean_g) * (std_r / std_g).clamp(0.75, 1.35) + mean_r
    mixed = generated * (1.0 - strength) + matched * strength
    return generated * (1.0 - valid) + mixed * valid


class KreaImageMaskedAspectPreserveRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "edit_mask": ("MASK",),
                "restore_map": ("KREA_IMAGE_ASPECT_MAP",),
                "mask_mode": (["manual if present, otherwise auto", "manual only", "auto difference"], {"default": "manual if present, otherwise auto"}),
                "difference_threshold": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005}),
                "difference_softness": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.2, "step": 0.001}),
                "mask_grow": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1}),
                "mask_feather": ("INT", {"default": 10, "min": 0, "max": 256, "step": 1}),
                "color_match_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("final_image", "restored_generated", "blend_mask", "restore_info")
    FUNCTION = "restore_masked"
    CATEGORY = "Krea 2/Native Resolution"

    def restore_masked(self, original_image, generated_image, edit_mask, restore_map,
                       mask_mode, difference_threshold, difference_softness,
                       mask_grow, mask_feather, color_match_strength):
        if original_image.ndim != 4 or generated_image.ndim != 4:
            raise ValueError("Expected IMAGE tensors as [B,H,W,C]")
        batch, target_h, target_w, channels = original_image.shape
        if batch != 1 or generated_image.shape[0] != 1:
            raise ValueError("This node currently expects one image")
        if target_w != restore_map["source_width"] or target_h != restore_map["source_height"]:
            raise ValueError("original_image dimensions do not match restore_map")

        out_device = original_image.device
        out_dtype = original_image.dtype
        work_device = _get_work_device(out_device)
        work_dtype = torch.float32

        print(f"[Krea Auto/Masked Restore] Starting on {work_device}: {target_w}x{target_h}")
        original = original_image.to(device=work_device, dtype=work_dtype)
        gen = generated_image.to(device=work_device, dtype=work_dtype)
        bucket_w = int(restore_map["bucket_width"])
        bucket_h = int(restore_map["bucket_height"])
        fitted_w = int(restore_map["fitted_width"])
        fitted_h = int(restore_map["fitted_height"])
        pad_left = int(restore_map["pad_left"])
        pad_top = int(restore_map["pad_top"])

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

        grid_x = (prepared_x / float(max(bucket_w - 1, 1))) * 2.0 - 1.0
        grid_y = (prepared_y / float(max(bucket_h - 1, 1))) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        restored = F.grid_sample(
            gen.permute(0, 3, 1, 2),
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        ).permute(0, 2, 3, 1)
        restored = torch.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        manual_mask = _prepare_mask(edit_mask, batch, target_h, target_w, work_device, work_dtype)
        manual_present = bool((manual_mask.max() > 0.01).item() and (manual_mask.mean() > 1e-6).item())

        # Remove global tone/color changes before measuring what was actually edited.
        mean_g = restored.mean(dim=(1, 2), keepdim=True)
        mean_o = original.mean(dim=(1, 2), keepdim=True)
        std_g = restored.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        std_o = original.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        detector = (restored - mean_g) * (std_o / std_g).clamp(0.80, 1.25) + mean_o
        detector = detector.clamp(0.0, 1.0)

        diff = (detector - original).abs().mean(dim=-1)
        # Suppress single-pixel upscaler noise while retaining coherent edits.
        diff = F.avg_pool2d(diff.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        softness = max(float(difference_softness), 1e-5)
        auto_mask = torch.sigmoid((diff - float(difference_threshold)) / softness)
        # Make truly unchanged pixels transparent instead of leaving a faint global veil.
        auto_mask = ((auto_mask - 0.15) / 0.85).clamp(0.0, 1.0)

        if mask_mode == "manual only":
            mask = manual_mask
            used_mode = "manual"
        elif mask_mode == "auto difference":
            mask = auto_mask
            used_mode = "auto"
        elif manual_present:
            mask = manual_mask
            used_mode = "manual"
        else:
            mask = auto_mask
            used_mode = "auto-fallback"

        restored = _local_color_match(restored, original, mask, color_match_strength).clamp(0.0, 1.0)
        alpha = _dilate(mask, int(mask_grow))
        alpha = _blur_mask(alpha, int(mask_feather)).clamp(0.0, 1.0)
        final = restored * alpha.unsqueeze(-1) + original * (1.0 - alpha.unsqueeze(-1))

        final = torch.nan_to_num(final, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if work_device.type == "cuda":
            torch.cuda.synchronize(work_device)

        final = final.to(device=out_device, dtype=out_dtype).contiguous()
        restored = restored.to(device=out_device, dtype=out_dtype).contiguous()
        alpha = alpha.to(device=out_device, dtype=out_dtype).contiguous()
        if out_device.type == "cuda":
            torch.cuda.synchronize(out_device)
        print(f"[Krea Auto/Masked Restore] Finished on {work_device}; mode={used_mode}; mask_mean={float(alpha.mean()):.4f}")

        info = (
            f"auto/masked native restore={target_w}x{target_h}; generated="
            f"{generated_image.shape[2]}x{generated_image.shape[1]}; "
            f"bucket={bucket_w}x{bucket_h}; mode={used_mode}; "
            f"manual_present={manual_present}; threshold={difference_threshold:.3f}; "
            f"softness={difference_softness:.3f}; mask_mean={float(alpha.mean()):.4f}; "
            f"mask_grow={mask_grow}; mask_feather={mask_feather}; "
            f"color_match={color_match_strength:.2f}; work_device={str(work_device)}"
        )
        return final, restored, alpha, info



class KreaImageUniversalAspectPreserveRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "edit_mask": ("MASK",),
                "restore_map": ("KREA_IMAGE_ASPECT_MAP",),
                "restore_mode": ([
                    "smart: manual mask, local auto, or full frame",
                    "full generated frame",
                    "manual mask only",
                    "auto local edit only",
                ], {"default": "smart: manual mask, local auto, or full frame"}),
                "difference_threshold": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005}),
                "difference_softness": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.2, "step": 0.001}),
                "max_local_coverage": ("FLOAT", {"default": 0.45, "min": 0.01, "max": 0.95, "step": 0.01}),
                "mask_grow": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1}),
                "mask_feather": ("INT", {"default": 10, "min": 0, "max": 256, "step": 1}),
                "local_color_match_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("final_image", "restored_generated", "blend_mask", "restore_info")
    FUNCTION = "restore_universal"
    CATEGORY = "Krea 2/Native Resolution"

    def restore_universal(self, original_image, generated_image, edit_mask, restore_map,
                          restore_mode, difference_threshold, difference_softness,
                          max_local_coverage, mask_grow, mask_feather,
                          local_color_match_strength):
        if original_image.ndim != 4 or generated_image.ndim != 4:
            raise ValueError("Expected IMAGE tensors as [B,H,W,C]")
        batch, target_h, target_w, channels = original_image.shape
        if batch != 1 or generated_image.shape[0] != 1:
            raise ValueError("This node currently expects one image")
        if target_w != restore_map["source_width"] or target_h != restore_map["source_height"]:
            raise ValueError("original_image dimensions do not match restore_map")

        out_device = original_image.device
        out_dtype = original_image.dtype
        work_device = _get_work_device(out_device)
        work_dtype = torch.float32

        print(f"[Krea Universal Restore] Starting on {work_device}: {target_w}x{target_h}")
        original = original_image.to(device=work_device, dtype=work_dtype)
        gen = generated_image.to(device=work_device, dtype=work_dtype)
        bucket_w = int(restore_map["bucket_width"])
        bucket_h = int(restore_map["bucket_height"])
        fitted_w = int(restore_map["fitted_width"])
        fitted_h = int(restore_map["fitted_height"])
        pad_left = int(restore_map["pad_left"])
        pad_top = int(restore_map["pad_top"])

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

        grid_x = (prepared_x / float(max(bucket_w - 1, 1))) * 2.0 - 1.0
        grid_y = (prepared_y / float(max(bucket_h - 1, 1))) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        restored = F.grid_sample(
            gen.permute(0, 3, 1, 2),
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        ).permute(0, 2, 3, 1)
        restored = torch.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        manual_mask = _prepare_mask(edit_mask, batch, target_h, target_w, work_device, work_dtype)
        manual_present = bool((manual_mask.max() > 0.01).item() and (manual_mask.mean() > 1e-6).item())

        # Difference detector removes broad global tone shifts so local edits can be isolated.
        mean_g = restored.mean(dim=(1, 2), keepdim=True)
        mean_o = original.mean(dim=(1, 2), keepdim=True)
        std_g = restored.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        std_o = original.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        detector = (restored - mean_g) * (std_o / std_g).clamp(0.80, 1.25) + mean_o
        detector = detector.clamp(0.0, 1.0)

        diff = (detector - original).abs().mean(dim=-1)
        diff = F.avg_pool2d(diff.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        softness = max(float(difference_softness), 1e-5)
        auto_mask = torch.sigmoid((diff - float(difference_threshold)) / softness)
        auto_mask = ((auto_mask - 0.15) / 0.85).clamp(0.0, 1.0)
        raw_auto_coverage = float((auto_mask > 0.5).float().mean().item())

        if restore_mode == "full generated frame":
            used_mode = "full-frame"
            alpha = torch.ones_like(manual_mask)
            final = restored
        elif restore_mode == "manual mask only":
            used_mode = "manual"
            alpha = manual_mask
            restored_local = _local_color_match(restored, original, alpha, local_color_match_strength).clamp(0.0, 1.0)
            alpha = _blur_mask(_dilate(alpha, int(mask_grow)), int(mask_feather)).clamp(0.0, 1.0)
            final = restored_local * alpha.unsqueeze(-1) + original * (1.0 - alpha.unsqueeze(-1))
        elif restore_mode == "auto local edit only":
            used_mode = "auto-local"
            alpha = auto_mask
            restored_local = _local_color_match(restored, original, alpha, local_color_match_strength).clamp(0.0, 1.0)
            alpha = _blur_mask(_dilate(alpha, int(mask_grow)), int(mask_feather)).clamp(0.0, 1.0)
            final = restored_local * alpha.unsqueeze(-1) + original * (1.0 - alpha.unsqueeze(-1))
        else:
            if manual_present:
                used_mode = "smart-manual"
                alpha = manual_mask
                restored_local = _local_color_match(restored, original, alpha, local_color_match_strength).clamp(0.0, 1.0)
                alpha = _blur_mask(_dilate(alpha, int(mask_grow)), int(mask_feather)).clamp(0.0, 1.0)
                final = restored_local * alpha.unsqueeze(-1) + original * (1.0 - alpha.unsqueeze(-1))
            elif 0.001 <= raw_auto_coverage <= float(max_local_coverage):
                used_mode = "smart-auto-local"
                alpha = auto_mask
                restored_local = _local_color_match(restored, original, alpha, local_color_match_strength).clamp(0.0, 1.0)
                alpha = _blur_mask(_dilate(alpha, int(mask_grow)), int(mask_feather)).clamp(0.0, 1.0)
                final = restored_local * alpha.unsqueeze(-1) + original * (1.0 - alpha.unsqueeze(-1))
            else:
                used_mode = "smart-full-frame"
                alpha = torch.ones_like(manual_mask)
                final = restored

        final = torch.nan_to_num(final, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if work_device.type == "cuda":
            torch.cuda.synchronize(work_device)

        final = final.to(device=out_device, dtype=out_dtype).contiguous()
        restored_out = restored.to(device=out_device, dtype=out_dtype).contiguous()
        alpha_out = alpha.to(device=out_device, dtype=out_dtype).contiguous()
        if out_device.type == "cuda":
            torch.cuda.synchronize(out_device)

        print(
            f"[Krea Universal Restore] Finished on {work_device}; mode={used_mode}; "
            f"manual={manual_present}; auto_coverage={raw_auto_coverage:.4f}"
        )
        info = (
            f"universal native restore={target_w}x{target_h}; generated="
            f"{generated_image.shape[2]}x{generated_image.shape[1]}; "
            f"bucket={bucket_w}x{bucket_h}; selected_mode={used_mode}; "
            f"manual_present={manual_present}; auto_coverage={raw_auto_coverage:.4f}; "
            f"max_local_coverage={float(max_local_coverage):.2f}; "
            f"threshold={float(difference_threshold):.3f}; softness={float(difference_softness):.3f}; "
            f"mask_grow={mask_grow}; mask_feather={mask_feather}; "
            f"local_color_match={float(local_color_match_strength):.2f}; "
            f"work_device={str(work_device)}"
        )
        return final, restored_out, alpha_out, info


NODE_CLASS_MAPPINGS = {
    "KreaImageAspectPreservePrepare": KreaImageAspectPreservePrepare,
    "KreaImageAspectPreserveRestore": KreaImageAspectPreserveRestore,
    "KreaImageMaskedAspectPreserveRestore": KreaImageMaskedAspectPreserveRestore,
    "KreaImageUniversalAspectPreserveRestore": KreaImageUniversalAspectPreserveRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KreaImageAspectPreservePrepare": "Krea Image Aspect-Preserve Prepare",
    "KreaImageAspectPreserveRestore": "Krea Image Aspect-Preserve Restore",
    "KreaImageMaskedAspectPreserveRestore": "Krea Image Auto/Masked Aspect-Preserve Restore",
    "KreaImageUniversalAspectPreserveRestore": "Krea Image Universal Aspect-Preserve Restore",
}
