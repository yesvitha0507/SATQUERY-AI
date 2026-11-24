from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}

MAX_SIZE = 50 * 1024 * 1024


def _extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_image(
    image_bytes: bytes,
    filename: str,
    content_type: str | None = None,
):
    """
    Validate an uploaded satellite/raster image.

    Supports:
    - PNG
    - JPG/JPEG
    - TIFF
    - GeoTIFF

    Returns useful metadata for the API/frontend.
    """

    if not filename:
        raise ValueError("Filename is missing.")

    if not image_bytes:
        raise ValueError("Image file is empty.")

    extension = _extension(filename)

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported format. Use PNG, JPG, JPEG, TIFF or GeoTIFF."
        )

    if len(image_bytes) > MAX_SIZE:
        raise ValueError(
            "Image must be smaller than 50 MB."
        )

    metadata = {
        "filename": filename,
        "extension": extension,
        "content_type": content_type,
        "size_bytes": len(image_bytes),
        "size_mb": round(
            len(image_bytes) / (1024 * 1024),
            2,
        ),
        "width": None,
        "height": None,
        "geospatial": False,
        "crs": None,
        "bounds": None,
        "transform": None,
        "resolution": None,
        "count": None,
        "dtype": None,
        "bands": [],
        "format": None,
        "mode": None,
    }

    # ============================================================
    # GEOTIFF / TIFF
    # ============================================================

    if extension in {".tif", ".tiff"}:

        # First try Rasterio.
        try:
            import rasterio

            with rasterio.MemoryFile(
                image_bytes
            ) as memfile:

                with memfile.open() as src:

                    metadata.update(
                        {
                            "width": src.width,
                            "height": src.height,
                            "format": "GeoTIFF"
                            if src.crs
                            else "TIFF",
                            "geospatial": bool(src.crs),
                            "crs": (
                                str(src.crs)
                                if src.crs
                                else None
                            ),
                            "bounds": [
                                float(src.bounds.left),
                                float(src.bounds.bottom),
                                float(src.bounds.right),
                                float(src.bounds.top),
                            ],
                            "transform": list(
                                src.transform
                            )[:6],
                            "resolution": [
                                float(src.res[0]),
                                float(src.res[1]),
                            ],
                            "count": src.count,
                            "dtype": (
                                str(src.dtypes[0])
                                if src.dtypes
                                else None
                            ),
                            "bands": [
                                (
                                    str(description)
                                    if description
                                    else f"Band {i + 1}"
                                )
                                for i, description
                                in enumerate(
                                    src.descriptions
                                )
                            ],
                        }
                    )

                    return metadata

        except Exception:
            # Rasterio may not be installed or
            # the TIFF may not be readable by Rasterio.
            pass

        # --------------------------------------------------------
        # Fallback: Pillow TIFF
        # --------------------------------------------------------

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            metadata.update(
                {
                    "width": image.width,
                    "height": image.height,
                    "format": image.format or "TIFF",
                    "mode": image.mode,
                    "count": len(
                        image.getbands()
                    ),
                }
            )

            image.verify()

            return metadata

        except Exception as error:

            raise ValueError(
                f"Unable to decode TIFF image: {error}"
            )

    # ============================================================
    # STANDARD IMAGE
    # ============================================================

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.verify()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = ImageOps.exif_transpose(
            image
        )

        metadata.update(
            {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "count": len(
                    image.getbands()
                ),
            }
        )

        return metadata

    except Exception as error:

        raise ValueError(
            f"Unable to decode image: {error}"
        )


def get_image_metadata(
    image_bytes: bytes,
    filename: str = "image.tif",
    content_type: str | None = "image/tiff",
):
    """
    Convenience wrapper.
    """

    return validate_image(
        image_bytes,
        filename,
        content_type,
    )