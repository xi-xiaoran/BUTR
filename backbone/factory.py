from __future__ import annotations
import torch.nn as nn
from .unet import UNet

# Optional backbones (some minimal code drops may not include them).
try:
    from .attention_unet import AttentionUNet  # type: ignore
except Exception:
    AttentionUNet = None  # type: ignore

try:
    from .unetpp import UNetPP  # type: ignore
except Exception:
    UNetPP = None  # type: ignore

try:
    from .segformer import SegFormerLite  # type: ignore
except Exception:
    SegFormerLite = None  # type: ignore

try:
    from .swin_unet import SwinUNetLite  # type: ignore
except Exception:
    SwinUNetLite = None  # type: ignore

# Accept common aliases so experiment scripts are forgiving.
_ALIASES = {
    # UNet
    "unet": "unet",
    # Attention-UNet
    "attentionunet": "attention_unet",
    "attention_unet": "attention_unet",
    "attention-unet": "attention_unet",
    "attnunet": "attention_unet",
    "attunet": "attention_unet",
    # UNet++
    "unetpp": "unetpp",
    "unet_pp": "unetpp",
    "unet++": "unetpp",
    # SegFormer (lite)
    "segformer": "segformer",
    "segformer_lite": "segformer",
    "segformer-lite": "segformer",
    # Swin-UNet (lite)
    "swinunet": "swin_unet",
    "swin_unet": "swin_unet",
    "swin-unet": "swin_unet",
    "swin_unet_lite": "swin_unet",
    "swinunet_lite": "swin_unet",
}

def build_backbone(name: str, head_mode: str = "standard", in_ch: int = 3, base: int = 32) -> nn.Module:
    name_in = (name or "").lower()
    name = _ALIASES.get(name_in, name_in)

    head_out_ch = 1 if head_mode == "standard" else 2

    if name == "unet":
        return UNet(in_ch=in_ch, base=base, head_out_ch=head_out_ch)
    if name == "attention_unet":
        if AttentionUNet is None:
            raise ValueError("attention_unet backbone is not included in this package")
        return AttentionUNet(in_ch=in_ch, base=base, head_out_ch=head_out_ch)
    if name == "unetpp":
        if UNetPP is None:
            raise ValueError("unetpp backbone is not included in this package")
        return UNetPP(in_ch=in_ch, base=base, head_out_ch=head_out_ch)
    if name == "segformer":
        if SegFormerLite is None:
            raise ValueError("segformer backbone is not included in this package")
        return SegFormerLite(in_ch=in_ch, head_out_ch=head_out_ch)
    if name == "swin_unet":
        if SwinUNetLite is None:
            raise ValueError("swin_unet backbone is not included in this package")
        return SwinUNetLite(in_ch=in_ch, base=base, head_out_ch=head_out_ch)

    raise ValueError(
        f"Unknown backbone: {name_in}. "
        f"Supported: unet, attention_unet(attunet/attnunet), unetpp(unet++), segformer, swin_unet(swinunet)."
    )
