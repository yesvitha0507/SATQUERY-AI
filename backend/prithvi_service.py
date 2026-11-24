from pathlib import Path

import numpy as np
import torch
import rasterio
from PIL import Image

from terratorch.registry import BACKBONE_REGISTRY


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


class PrithviService:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading Prithvi-EO-2.0 300M...")

        self.model = BACKBONE_REGISTRY.build(
            "terratorch_prithvi_eo_v2_300"
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        print("Prithvi loaded successfully!")
        print(f"Device: {self.device}")

    # ---------------------------------------------------------
    # VALIDATE FILE
    # ---------------------------------------------------------

    def validate_file(self, image_path):

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}"
            )

    # ---------------------------------------------------------
    # LOAD IMAGE
    # ---------------------------------------------------------

    def load_image(self, image_path):

        self.validate_file(image_path)

        path = Path(image_path)
        extension = path.suffix.lower()

        # =====================================================
        # TIFF / GEOTIFF
        # =====================================================

        if extension in [".tif", ".tiff"]:

            with rasterio.open(image_path) as src:

                image = src.read()

                metadata = {
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "crs": str(src.crs) if src.crs else None,
                    "is_geotiff": src.crs is not None,
                    "resolution": src.res,
                    "bounds": [
                        src.bounds.left,
                        src.bounds.bottom,
                        src.bounds.right,
                        src.bounds.top,
                    ],
                }

        # =====================================================
        # JPG / JPEG / PNG
        # =====================================================

        else:

            img = Image.open(image_path).convert("RGB")

            image = np.array(img)

            # HWC → CHW
            image = np.transpose(image, (2, 0, 1))

            metadata = {
                "width": img.width,
                "height": img.height,
                "bands": 3,
                "crs": None,
                "is_geotiff": False,
                "resolution": None,
                "bounds": None,
            }

        # =====================================================
        # CLEAN DATA
        # =====================================================

        image = image.astype(np.float32)

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return image, metadata

    # ---------------------------------------------------------
    # IMAGE INFORMATION
    # ---------------------------------------------------------

    def inspect_image(self, image_path):

        image, metadata = self.load_image(image_path)

        return {
            "shape": list(image.shape),
            "width": metadata["width"],
            "height": metadata["height"],
            "bands": metadata["bands"],
            "crs": metadata["crs"],
            "is_geotiff": metadata["is_geotiff"],
            "resolution": metadata["resolution"],
            "bounds": metadata["bounds"],
        }