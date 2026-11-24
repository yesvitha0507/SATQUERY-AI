from __future__ import annotations

import io
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw


# ============================================================
# GENERAL HELPERS
# ============================================================

def _normalize_band(
    arr: np.ndarray,
) -> np.ndarray:

    arr = np.asarray(
        arr
    ).astype(
        np.float32
    )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if arr.size == 0:
        return np.zeros_like(
            arr,
            dtype=np.float32,
        )

    low, high = np.percentile(
        arr,
        [2, 98],
    )

    if high <= low:

        low = float(
            arr.min()
        )

        high = float(
            arr.max()
        )

    if high <= low:
        return np.zeros_like(
            arr,
            dtype=np.float32,
        )

    return np.clip(
        (arr - low)
        /
        (high - low),
        0,
        1,
    )


def _to_uint8(
    arr: np.ndarray,
) -> np.ndarray:

    return (
        np.clip(
            arr,
            0,
            1,
        )
        * 255
    ).astype(
        np.uint8
    )


def _safe_float(
    value,
    default=0.0,
):

    try:

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):
            return default

        return value

    except Exception:

        return default


def _safe_round(
    value,
    digits=4,
):

    return round(
        _safe_float(value),
        digits,
    )


def _compressed_values(
    masked_array,
):

    try:

        values = (
            masked_array
            .compressed()
        )

    except Exception:

        values = np.asarray(
            masked_array
        ).reshape(-1)

    values = np.asarray(
        values
    )

    values = values[
        np.isfinite(values)
    ]

    return values


# ============================================================
# BAND STATISTICS
# ============================================================

def _band_statistics(
    band,
) -> dict:

    band = np.ma.masked_invalid(
        np.asarray(
            band
        ).astype(
            np.float32
        )
    )

    values = _compressed_values(
        band
    )

    if values.size == 0:

        return {
            "mean": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "p10": 0,
            "p25": 0,
            "median": 0,
            "p75": 0,
            "p90": 0,
        }

    return {
        "mean": _safe_round(
            values.mean()
        ),
        "std": _safe_round(
            values.std()
        ),
        "min": _safe_round(
            values.min()
        ),
        "max": _safe_round(
            values.max()
        ),
        "p10": _safe_round(
            np.percentile(
                values,
                10,
            )
        ),
        "p25": _safe_round(
            np.percentile(
                values,
                25,
            )
        ),
        "median": _safe_round(
            np.median(values)
        ),
        "p75": _safe_round(
            np.percentile(
                values,
                75,
            )
        ),
        "p90": _safe_round(
            np.percentile(
                values,
                90,
            )
        ),
    }


# ============================================================
# INDEX STATISTICS
# ============================================================

def _index_statistics(
    index,
) -> dict:

    index = np.ma.masked_invalid(
        index
    )

    values = _compressed_values(
        index
    )

    if values.size == 0:

        return {
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "p10": 0,
            "p90": 0,
        }

    return {
        "mean": _safe_round(
            values.mean()
        ),
        "median": _safe_round(
            np.median(values)
        ),
        "std": _safe_round(
            values.std()
        ),
        "min": _safe_round(
            values.min()
        ),
        "max": _safe_round(
            values.max()
        ),
        "p10": _safe_round(
            np.percentile(
                values,
                10,
            )
        ),
        "p90": _safe_round(
            np.percentile(
                values,
                90,
            )
        ),
    }


# ============================================================
# GEOSPATIAL METRICS
# ============================================================

def _calculate_geospatial_metrics(
    src,
) -> dict:

    result = {
        "crs": (
            str(src.crs)
            if src.crs
            else None
        ),

        "bounds": [
            _safe_float(
                src.bounds.left
            ),
            _safe_float(
                src.bounds.bottom
            ),
            _safe_float(
                src.bounds.right
            ),
            _safe_float(
                src.bounds.top
            ),
        ],

        "resolution": [
            _safe_float(
                src.res[0]
            ),
            _safe_float(
                src.res[1]
            ),
        ],

        "pixel_width": _safe_float(
            src.res[0]
        ),

        "pixel_height": _safe_float(
            src.res[1]
        ),

        "width_pixels": int(
            src.width
        ),

        "height_pixels": int(
            src.height
        ),

        "coordinate_system_type": None,

        "center": {
            "x": _safe_float(
                (
                    src.bounds.left
                    + src.bounds.right
                )
                / 2
            ),
            "y": _safe_float(
                (
                    src.bounds.bottom
                    + src.bounds.top
                )
                / 2
            ),
        },

        "scene_area": None,
    }

    # --------------------------------------------------------
    # CRS TYPE
    # --------------------------------------------------------

    try:

        if src.crs:

            if src.crs.is_projected:

                result[
                    "coordinate_system_type"
                ] = "projected"

            elif src.crs.is_geographic:

                result[
                    "coordinate_system_type"
                ] = "geographic"

    except Exception:
        pass

    # --------------------------------------------------------
    # PROJECTED AREA
    # --------------------------------------------------------

    try:

        if (
            src.crs
            and src.crs.is_projected
        ):

            pixel_area = abs(
                float(src.res[0])
                * float(src.res[1])
            )

            scene_area_m2 = (
                src.width
                * src.height
                * pixel_area
            )

            result[
                "scene_area"
            ] = {
                "square_meters": round(
                    scene_area_m2,
                    2,
                ),

                "square_kilometers": round(
                    scene_area_m2
                    / 1_000_000,
                    6,
                ),

                "pixel_area_square_meters": round(
                    pixel_area,
                    4,
                ),

                "method": (
                    "pixel count × projected pixel area"
                ),
            }

    except Exception:
        pass

    # --------------------------------------------------------
    # WGS84 CONVERSION
    # --------------------------------------------------------

    try:

        if src.crs:

            from pyproj import Transformer

            transformer = (
                Transformer.from_crs(
                    src.crs,
                    "EPSG:4326",
                    always_xy=True,
                )
            )

            lon, lat = transformer.transform(
                result["center"]["x"],
                result["center"]["y"],
            )

            result[
                "center_latitude_longitude"
            ] = {
                "latitude": round(
                    float(lat),
                    6,
                ),
                "longitude": round(
                    float(lon),
                    6,
                ),
            }

            left, bottom = (
                transformer.transform(
                    src.bounds.left,
                    src.bounds.bottom,
                )
            )

            right, top = (
                transformer.transform(
                    src.bounds.right,
                    src.bounds.top,
                )
            )

            result[
                "wgs84_bounds"
            ] = [
                round(
                    float(left),
                    6,
                ),
                round(
                    float(bottom),
                    6,
                ),
                round(
                    float(right),
                    6,
                ),
                round(
                    float(top),
                    6,
                ),
            ]

    except Exception:

        result[
            "coordinate_conversion"
        ] = "Unavailable"

    return result


# ============================================================
# SENTINEL-2 BAND IDENTIFICATION
# ============================================================

def _identify_sentinel_bands(
    src,
) -> dict:

    descriptions = [
        (
            desc or ""
        ).upper()
        for desc in src.descriptions
    ]

    mapping = {}

    aliases = {
        "B01": 1,
        "B02": 2,
        "B03": 3,
        "B04": 4,
        "B05": 5,
        "B06": 6,
        "B07": 7,
        "B08": 8,
        "B8A": 9,
        "B09": 10,
        "B11": 11,
        "B12": 12,
    }

    for index, description in enumerate(
        descriptions
    ):

        for band_name in aliases:

            if band_name in description:

                mapping[
                    band_name
                ] = index

    # --------------------------------------------------------
    # Fallback for canonical 12-band ordering.
    # --------------------------------------------------------

    if (
        len(mapping) < 4
        and src.count >= 12
    ):

        mapping = {
            "B01": 0,
            "B02": 1,
            "B03": 2,
            "B04": 3,
            "B05": 4,
            "B06": 5,
            "B07": 6,
            "B08": 7,
            "B8A": 8,
            "B09": 9,
            "B11": 10,
            "B12": 11,
        }

    return mapping


# ============================================================
# RASTER PREVIEW
# ============================================================

def _pil_preview_from_rasterio(
    src,
) -> tuple[Image.Image, dict]:

    count = src.count

    height = src.height
    width = src.width

    data = src.read()

    band_map = _identify_sentinel_bands(
        src
    )

    # --------------------------------------------------------
    # Sentinel-2 RGB
    # --------------------------------------------------------

    if (
        "B04" in band_map
        and "B03" in band_map
        and "B02" in band_map
    ):

        rgb = np.stack(
            [
                _normalize_band(
                    data[
                        band_map["B04"]
                    ]
                ),
                _normalize_band(
                    data[
                        band_map["B03"]
                    ]
                ),
                _normalize_band(
                    data[
                        band_map["B02"]
                    ]
                ),
            ],
            axis=-1,
        )

        preview_type = (
            "Sentinel-2 true-colour RGB"
        )

    # --------------------------------------------------------
    # Generic RGB
    # --------------------------------------------------------

    elif count >= 3:

        rgb = np.stack(
            [
                _normalize_band(
                    data[0]
                ),
                _normalize_band(
                    data[1]
                ),
                _normalize_band(
                    data[2]
                ),
            ],
            axis=-1,
        )

        preview_type = (
            "Generic RGB composite"
        )

    # --------------------------------------------------------
    # Single band
    # --------------------------------------------------------

    else:

        gray = _normalize_band(
            data[0]
        )

        rgb = np.repeat(
            gray[..., None],
            3,
            axis=-1,
        )

        preview_type = (
            "Single-band grayscale preview"
        )

    image = Image.fromarray(
        _to_uint8(rgb),
        mode="RGB",
    )

    image = ImageOps.exif_transpose(
        image
    )

    original_size = image.size

    image.thumbnail(
        (1400, 1000)
    )

    return (
        image,
        {
            "width": width,
            "height": height,
            "count": count,
            "dtype": str(
                data.dtype
            ),
            "original_preview_size": list(
                original_size
            ),
            "preview_size": list(
                image.size
            ),
            "preview_type": preview_type,
        },
    )


# ============================================================
# CREATE PREVIEW
# ============================================================

def create_preview(
    image_bytes: bytes,
    filename: str,
    output_dir: Path,
) -> tuple[bytes, dict]:

    output_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    extension = Path(
        filename
    ).suffix.lower()

    # ========================================================
    # TIFF / GEOTIFF
    # ========================================================

    if extension in {
        ".tif",
        ".tiff",
    }:

        try:

            import rasterio

            with rasterio.MemoryFile(
                image_bytes
            ) as memfile:

                with memfile.open() as src:

                    image, stats = (
                        _pil_preview_from_rasterio(
                            src
                        )
                    )

                    metadata = {
                        "source_type": (
                            "GeoTIFF"
                            if src.crs
                            else "TIFF"
                        ),

                        "geospatial": bool(
                            src.crs
                        ),

                        "crs": (
                            str(src.crs)
                            if src.crs
                            else None
                        ),

                        "bounds": [
                            _safe_float(
                                src.bounds.left
                            ),
                            _safe_float(
                                src.bounds.bottom
                            ),
                            _safe_float(
                                src.bounds.right
                            ),
                            _safe_float(
                                src.bounds.top
                            ),
                        ],

                        "resolution": [
                            _safe_float(
                                src.res[0]
                            ),
                            _safe_float(
                                src.res[1]
                            ),
                        ],

                        "count": src.count,

                        "width": src.width,

                        "height": src.height,

                        "dtype": (
                            str(
                                src.dtypes[0]
                            )
                            if src.dtypes
                            else None
                        ),

                        "bands": [
                            (
                                desc
                                if desc
                                else f"Band {i + 1}"
                            )
                            for i, desc in enumerate(
                                src.descriptions
                            )
                        ],

                        **stats,
                    }

                    output = io.BytesIO()

                    image.save(
                        output,
                        format="JPEG",
                        quality=88,
                        optimize=True,
                    )

                    preview_name = (
                        f"{uuid.uuid4().hex}"
                        "_preview.jpg"
                    )

                    (
                        output_dir
                        / preview_name
                    ).write_bytes(
                        output.getvalue()
                    )

                    return (
                        output.getvalue(),
                        {
                            **metadata,
                            "preview_name": (
                                preview_name
                            ),
                        },
                    )

        except Exception:
            pass

    # ========================================================
    # STANDARD IMAGE
    # ========================================================

    try:

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        original_format = (
            image.format
        )

        image = (
            ImageOps
            .exif_transpose(image)
            .convert("RGB")
        )

        original_width = (
            image.width
        )

        original_height = (
            image.height
        )

        image.thumbnail(
            (1400, 1000)
        )

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=88,
            optimize=True,
        )

        preview_name = (
            f"{uuid.uuid4().hex}"
            "_preview.jpg"
        )

        (
            output_dir
            / preview_name
        ).write_bytes(
            output.getvalue()
        )

        return (
            output.getvalue(),
            {
                "source_type": (
                    original_format
                    or "Raster"
                ),

                "geospatial": False,

                "width": original_width,

                "height": original_height,

                "count": 3,

                "preview_size": list(
                    image.size
                ),

                "preview_name": preview_name,
            },
        )

    except Exception as error:

        raise ValueError(
            f"Unable to create image preview: {error}"
        )


# ============================================================
# RASTER ANALYSIS
# ============================================================

def raster_analysis(
    image_bytes: bytes,
    filename: str,
) -> dict:

    extension = Path(
        filename
    ).suffix.lower()

    result = {
        "filename": filename,
        "sensor_hints": [],
        "indices": {},
        "index_statistics": {},
        "statistics": {},
        "geospatial": {},
        "semantic_indicators": {},
        "band_descriptions": [],
    }

    # ========================================================
    # TIFF / GEOTIFF
    # ========================================================

    if extension in {
        ".tif",
        ".tiff",
    }:

        try:

            import rasterio

            with rasterio.MemoryFile(
                image_bytes
            ) as memfile:

                with memfile.open() as src:

                    data = src.read(
                        masked=True
                    ).astype(
                        np.float32
                    )

                    count = src.count

                    # ------------------------------------------------
                    # Geospatial
                    # ------------------------------------------------

                    result[
                        "geospatial"
                    ] = (
                        _calculate_geospatial_metrics(
                            src
                        )
                    )

                    result[
                        "geospatial"
                    ].update(
                        {
                            "band_count": count,
                            "width": src.width,
                            "height": src.height,
                            "dtype": (
                                str(
                                    src.dtypes[0]
                                )
                                if src.dtypes
                                else None
                            ),
                        }
                    )

                    # ------------------------------------------------
                    # Band descriptions
                    # ------------------------------------------------

                    result[
                        "band_descriptions"
                    ] = [
                        (
                            desc
                            if desc
                            else f"Band {i + 1}"
                        )
                        for i, desc in enumerate(
                            src.descriptions
                        )
                    ]

                    # ------------------------------------------------
                    # Identify Sentinel bands
                    # ------------------------------------------------

                    sentinel = (
                        _identify_sentinel_bands(
                            src
                        )
                    )

                    # =================================================
                    # SENTINEL / MULTISPECTRAL
                    # =================================================

                    if (
                        "B02" in sentinel
                        and "B03" in sentinel
                        and "B04" in sentinel
                        and "B08" in sentinel
                    ):

                        result[
                            "sensor_hints"
                        ].append(
                            "Sentinel-2-compatible spectral band mapping detected"
                        )

                        blue = data[
                            sentinel["B02"]
                        ]

                        green = data[
                            sentinel["B03"]
                        ]

                        red = data[
                            sentinel["B04"]
                        ]

                        nir = data[
                            sentinel["B08"]
                        ]

                        # ------------------------------------------------
                        # NDVI
                        # ------------------------------------------------

                        ndvi = (
                            nir - red
                        ) / (
                            nir
                            + red
                            + 1e-6
                        )

                        ndvi = (
                            np.ma.masked_invalid(
                                ndvi
                            )
                        )

                        result[
                            "indices"
                        ][
                            "NDVI"
                        ] = _safe_round(
                            ndvi.mean()
                        )

                        result[
                            "index_statistics"
                        ][
                            "NDVI"
                        ] = (
                            _index_statistics(
                                ndvi
                            )
                        )

                        # ------------------------------------------------
                        # NDWI
                        # ------------------------------------------------

                        ndwi = (
                            green - nir
                        ) / (
                            green
                            + nir
                            + 1e-6
                        )

                        ndwi = (
                            np.ma.masked_invalid(
                                ndwi
                            )
                        )

                        result[
                            "indices"
                        ][
                            "NDWI"
                        ] = _safe_round(
                            ndwi.mean()
                        )

                        result[
                            "index_statistics"
                        ][
                            "NDWI"
                        ] = (
                            _index_statistics(
                                ndwi
                            )
                        )

                        # ------------------------------------------------
                        # NDBI
                        # ------------------------------------------------

                        if "B11" in sentinel:

                            swir = data[
                                sentinel["B11"]
                            ]

                        elif "B12" in sentinel:

                            swir = data[
                                sentinel["B12"]
                            ]

                        else:

                            swir = None

                        if swir is not None:

                            ndbi = (
                                swir - nir
                            ) / (
                                swir
                                + nir
                                + 1e-6
                            )

                            ndbi = (
                                np.ma.masked_invalid(
                                    ndbi
                                )
                            )

                            result[
                                "indices"
                            ][
                                "NDBI"
                            ] = _safe_round(
                                ndbi.mean()
                            )

                            result[
                                "index_statistics"
                            ][
                                "NDBI"
                            ] = (
                                _index_statistics(
                                    ndbi
                                )
                            )

                        # ------------------------------------------------
                        # Semantic indicators
                        # ------------------------------------------------

                        valid_ndvi = (
                            _compressed_values(
                                ndvi
                            )
                        )

                        valid_ndwi = (
                            _compressed_values(
                                ndwi
                            )
                        )

                        if valid_ndvi.size:

                            result[
                                "semantic_indicators"
                            ][
                                "vegetation_fraction_ndvi_gt_0_3"
                            ] = round(
                                float(
                                    np.mean(
                                        valid_ndvi
                                        > 0.3
                                    )
                                ),
                                4,
                            )

                            result[
                                "semantic_indicators"
                            ][
                                "dense_vegetation_fraction_ndvi_gt_0_5"
                            ] = round(
                                float(
                                    np.mean(
                                        valid_ndvi
                                        > 0.5
                                    )
                                ),
                                4,
                            )

                        if valid_ndwi.size:

                            result[
                                "semantic_indicators"
                            ][
                                "water_likelihood_ndwi_gt_0_2"
                            ] = round(
                                float(
                                    np.mean(
                                        valid_ndwi
                                        > 0.2
                                    )
                                ),
                                4,
                            )

                        if swir is not None:

                            valid_ndbi = (
                                _compressed_values(
                                    ndbi
                                )
                            )

                            if valid_ndbi.size:

                                result[
                                    "semantic_indicators"
                                ][
                                    "built_up_likelihood_ndbi_gt_0"
                                ] = round(
                                    float(
                                        np.mean(
                                            valid_ndbi
                                            > 0
                                        )
                                    ),
                                    4,
                                )

                        # ------------------------------------------------
                        # Statistics
                        # ------------------------------------------------

                        band_names = {
                            "B02_blue": "B02",
                            "B03_green": "B03",
                            "B04_red": "B04",
                            "B08_nir": "B08",
                        }

                        if "B11" in sentinel:

                            band_names[
                                "B11_swir"
                            ] = "B11"

                        if "B12" in sentinel:

                            band_names[
                                "B12_swir"
                            ] = "B12"

                        for name, band_name in (
                            band_names.items()
                        ):

                            result[
                                "statistics"
                            ][
                                name
                            ] = (
                                _band_statistics(
                                    data[
                                        sentinel[
                                            band_name
                                        ]
                                    ]
                                )
                            )

                    # =================================================
                    # TWO BAND SAR
                    # =================================================

                    elif count == 2:

                        result[
                            "sensor_hints"
                        ].append(
                            "Two-band raster; potentially dual-polarization SAR (VV/VH)"
                        )

                        for i, name in enumerate(
                            [
                                "Band1",
                                "Band2",
                            ]
                        ):

                            result[
                                "statistics"
                            ][
                                name
                            ] = (
                                _band_statistics(
                                    data[i]
                                )
                            )

                        try:

                            vv = data[0]
                            vh = data[1]

                            ratio = (
                                np.abs(vv)
                                /
                                (
                                    np.abs(vh)
                                    + 1e-6
                                )
                            )

                            ratio = (
                                np.ma.masked_invalid(
                                    ratio
                                )
                            )

                            result[
                                "sar_analysis"
                            ] = {
                                "vv_vh_ratio_mean": (
                                    _safe_round(
                                        ratio.mean()
                                    )
                                ),

                                "interpretation": (
                                    "Two-band backscatter "
                                    "structure may correspond "
                                    "to dual-polarization SAR."
                                ),
                            }

                        except Exception:
                            pass

                    # =================================================
                    # SINGLE BAND
                    # =================================================

                    elif count == 1:

                        result[
                            "sensor_hints"
                        ].append(
                            "Single-band raster; compatible with grayscale or SAR-style analysis"
                        )

                        result[
                            "statistics"
                        ][
                            "single_band"
                        ] = (
                            _band_statistics(
                                data[0]
                            )
                        )

                    # =================================================
                    # OTHER MULTISPECTRAL
                    # =================================================

                    else:

                        result[
                            "sensor_hints"
                        ].append(
                            f"{count}-band multispectral/hyperspectral raster"
                        )

                        for i in range(
                            min(
                                count,
                                20,
                            )
                        ):

                            result[
                                "statistics"
                            ][
                                f"Band_{i + 1}"
                            ] = (
                                _band_statistics(
                                    data[i]
                                )
                            )

                    return result

        except Exception as error:

            result[
                "raster_error"
            ] = (
                f"{type(error).__name__}: "
                f"{str(error)}"
            )

    # ========================================================
    # STANDARD RGB
    # ========================================================

    try:

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert(
            "RGB"
        )

        arr = (
            np.asarray(
                image
            ).astype(
                np.float32
            )
            / 255.0
        )

        result[
            "statistics"
        ] = {
            "mean_rgb": [
                _safe_round(
                    arr[
                        ...,
                        i
                    ].mean()
                )
                for i in range(3)
            ],

            "std_rgb": [
                _safe_round(
                    arr[
                        ...,
                        i
                    ].std()
                )
                for i in range(3)
            ],
        }

        result[
            "geospatial"
        ] = {
            "geospatial": False,
            "width": image.width,
            "height": image.height,
        }

        result[
            "sensor_hints"
        ].append(
            "RGB image; spectral indices require multispectral bands"
        )

    except Exception:
        pass

    return result


# ============================================================
# TEMPORAL IMAGE PREPARATION
# ============================================================

def _prepare_comparison_images(
    before_bytes: bytes,
    before_filename: str,
    after_bytes: bytes,
    after_filename: str,
    output_dir: Path,
):

    before_preview, before_meta = (
        create_preview(
            before_bytes,
            before_filename,
            output_dir,
        )
    )

    after_preview, after_meta = (
        create_preview(
            after_bytes,
            after_filename,
            output_dir,
        )
    )

    before = Image.open(
        io.BytesIO(
            before_preview
        )
    ).convert(
        "RGB"
    )

    after = Image.open(
        io.BytesIO(
            after_preview
        )
    ).convert(
        "RGB"
    )

    width = min(
        before.width,
        after.width,
    )

    height = min(
        before.height,
        after.height,
    )

    before = before.resize(
        (
            width,
            height,
        )
    )

    after = after.resize(
        (
            width,
            height,
        )
    )

    return (
        before,
        after,
        before_meta,
        after_meta,
    )


# ============================================================
# CHANGE DETECTION
# ============================================================

def create_change_map(
    before_bytes: bytes,
    before_filename: str,
    after_bytes: bytes,
    after_filename: str,
    output_dir: Path,
) -> tuple[str, dict]:

    (
        before,
        after,
        before_meta,
        after_meta,
    ) = _prepare_comparison_images(
        before_bytes,
        before_filename,
        after_bytes,
        after_filename,
        output_dir,
    )

    a = (
        np.asarray(
            before
        ).astype(
            np.float32
        )
        / 255.0
    )

    b = (
        np.asarray(
            after
        ).astype(
            np.float32
        )
        / 255.0
    )

    # ========================================================
    # DIFFERENCE
    # ========================================================

    diff = np.mean(
        np.abs(
            a - b
        ),
        axis=2,
    )

    # ========================================================
    # ADAPTIVE THRESHOLD
    # ========================================================

    threshold = float(
        np.percentile(
            diff,
            85,
        )
    )

    threshold = max(
        threshold,
        0.05,
    )

    change = (
        diff >= threshold
    )

    # ========================================================
    # MORPHOLOGICAL CLEANUP
    # ========================================================

    components = []

    try:

        import cv2

        mask = (
            change.astype(
                np.uint8
            )
            * 255
        )

        kernel = np.ones(
            (
                5,
                5,
            ),
            np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        change = (
            mask > 0
        )

        (
            num_labels,
            labels,
            stats,
            centroids,
        ) = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        for i in range(
            1,
            num_labels,
        ):

            area = int(
                stats[
                    i,
                    cv2.CC_STAT_AREA,
                ]
            )

            if area < 25:
                continue

            x = int(
                stats[
                    i,
                    cv2.CC_STAT_LEFT,
                ]
            )

            y = int(
                stats[
                    i,
                    cv2.CC_STAT_TOP,
                ]
            )

            w = int(
                stats[
                    i,
                    cv2.CC_STAT_WIDTH,
                ]
            )

            h = int(
                stats[
                    i,
                    cv2.CC_STAT_HEIGHT,
                ]
            )

            cx, cy = (
                centroids[i]
            )

            components.append(
                {
                    "area_pixels": area,
                    "bounding_box": [
                        x,
                        y,
                        w,
                        h,
                    ],
                    "centroid_pixel": [
                        round(
                            float(cx),
                            2,
                        ),
                        round(
                            float(cy),
                            2,
                        ),
                    ],
                }
            )

        components.sort(
            key=lambda x: x[
                "area_pixels"
            ],
            reverse=True,
        )

    except Exception:
        pass

    # ========================================================
    # CHANGE METRICS
    # ========================================================

    changed_pixels = int(
        np.sum(change)
    )

    total_pixels = int(
        change.size
    )

    change_ratio = (
        changed_pixels
        /
        max(
            total_pixels,
            1,
        )
    )

    # ========================================================
    # SPATIAL DISTRIBUTION
    # ========================================================

    height, width = (
        change.shape
    )

    spatial_distribution = {}

    regions = {
        "upper_left": (
            0,
            height // 2,
            0,
            width // 2,
        ),

        "upper_right": (
            0,
            height // 2,
            width // 2,
            width,
        ),

        "lower_left": (
            height // 2,
            height,
            0,
            width // 2,
        ),

        "lower_right": (
            height // 2,
            height,
            width // 2,
            width,
        ),

        "center": (
            height // 4,
            3 * height // 4,
            width // 4,
            3 * width // 4,
        ),
    }

    for name, (
        y1,
        y2,
        x1,
        x2,
    ) in regions.items():

        region = change[
            y1:y2,
            x1:x2,
        ]

        spatial_distribution[
            name
        ] = round(
            float(
                region.mean()
            ),
            4,
        )

    # ========================================================
    # GEOSPATIAL CHANGE
    # ========================================================

    geospatial_change = {
        "available": False,
    }

    try:

        if (
            before_meta.get(
                "geospatial"
            )
            and before_meta.get(
                "resolution"
            )
        ):

            pixel_width = abs(
                float(
                    before_meta[
                        "resolution"
                    ][0]
                )
            )

            pixel_height = abs(
                float(
                    before_meta[
                        "resolution"
                    ][1]
                )
            )

            pixel_area = (
                pixel_width
                * pixel_height
            )

            changed_area_m2 = (
                changed_pixels
                * pixel_area
            )

            geospatial_change = {
                "available": True,

                "pixel_width_m": round(
                    pixel_width,
                    4,
                ),

                "pixel_height_m": round(
                    pixel_height,
                    4,
                ),

                "pixel_area_m2": round(
                    pixel_area,
                    4,
                ),

                "changed_area_m2": round(
                    changed_area_m2,
                    2,
                ),

                "changed_area_hectares": round(
                    changed_area_m2
                    / 10_000,
                    4,
                ),

                "changed_area_km2": round(
                    changed_area_m2
                    / 1_000_000,
                    6,
                ),

                "crs": before_meta.get(
                    "crs"
                ),

                "method": (
                    "changed pixel count × "
                    "GeoTIFF pixel area"
                ),
            }

    except Exception:
        pass

    # ========================================================
    # CREATE EVIDENCE IMAGE
    # ========================================================

    overlay = np.asarray(
        after
    ).copy()

    overlay[
        change
    ] = np.array(
        [
            255,
            60,
            80,
        ],
        dtype=np.uint8,
    )

    base_img = np.asarray(
        after
    ).copy()

    overlay_img = (
        0.55 * base_img
        +
        0.45 * overlay
    ).astype(
        np.uint8
    )

    evidence = Image.fromarray(
        overlay_img
    )

    draw = ImageDraw.Draw(
        evidence
    )

    draw.rectangle(
        (
            10,
            10,
            440,
            75,
        ),
        fill=(
            5,
            7,
            11,
        ),
    )

    draw.text(
        (
            20,
            20,
        ),
        (
            f"CHANGE MAP | "
            f"threshold={threshold:.3f}"
        ),
        fill=(
            255,
            255,
            255,
        ),
    )

    draw.text(
        (
            20,
            42,
        ),
        (
            f"Changed area: "
            f"{change_ratio * 100:.2f}%"
        ),
        fill=(
            255,
            255,
            255,
        ),
    )

    filename = (
        f"{uuid.uuid4().hex}"
        "_change_map.jpg"
    )

    path = (
        output_dir
        / filename
    )

    evidence.save(
        path,
        format="JPEG",
        quality=88,
    )

    return (
        f"/generated/{filename}",
        {
            "threshold": round(
                threshold,
                4,
            ),

            "changed_pixels": (
                changed_pixels
            ),

            "total_pixels": (
                total_pixels
            ),

            "changed_pixel_ratio": round(
                change_ratio,
                4,
            ),

            "changed_area_percent": round(
                change_ratio * 100,
                2,
            ),

            "spatial_distribution": (
                spatial_distribution
            ),

            "change_components": (
                components[:20]
            ),

            "geospatial_change": (
                geospatial_change
            ),

            "method": (
                "image difference + adaptive "
                "percentile threshold + morphological cleanup"
            ),

            "warning": (
                "Highlighted pixels represent "
                "spectral/visual difference and should "
                "not automatically be interpreted as "
                "semantic land-cover change."
            ),
        },
    )


# ============================================================
# OPTICAL + SAR FUSION
# ============================================================

def create_optical_sar_evidence(
    optical_bytes: bytes,
    optical_filename: str,
    sar_bytes: bytes,
    sar_filename: str,
    output_dir: Path,
) -> tuple[str, dict]:

    optical_preview, optical_meta = (
        create_preview(
            optical_bytes,
            optical_filename,
            output_dir,
        )
    )

    sar_preview, sar_meta = (
        create_preview(
            sar_bytes,
            sar_filename,
            output_dir,
        )
    )

    optical = Image.open(
        io.BytesIO(
            optical_preview
        )
    ).convert(
        "RGB"
    )

    sar = Image.open(
        io.BytesIO(
            sar_preview
        )
    ).convert(
        "RGB"
    )

    width = min(
        optical.width,
        sar.width,
    )

    height = min(
        optical.height,
        sar.height,
    )

    optical = optical.resize(
        (
            width,
            height,
        )
    )

    sar = sar.resize(
        (
            width,
            height,
        )
    )

    o = (
        np.asarray(
            optical
        ).astype(
            np.float32
        )
        / 255.0
    )

    s = (
        np.asarray(
            sar
        ).astype(
            np.float32
        )
        / 255.0
    )

    optical_intensity = (
        o.mean(axis=2)
    )

    sar_intensity = (
        s.mean(axis=2)
    )

    # ========================================================
    # FUSION
    # ========================================================

    fused = np.zeros_like(
        o
    )

    fused[..., 0] = (
        optical_intensity
    )

    fused[..., 1] = (
        0.65
        * optical_intensity
        +
        0.35
        * sar_intensity
    )

    fused[..., 2] = (
        sar_intensity
    )

    fused = _to_uint8(
        fused
    )

    evidence = Image.fromarray(
        fused
    )

    draw = ImageDraw.Draw(
        evidence
    )

    draw.rectangle(
        (
            10,
            10,
            410,
            55,
        ),
        fill=(
            5,
            7,
            11,
        ),
    )

    draw.text(
        (
            20,
            20,
        ),
        "OPTICAL + SAR FUSION EVIDENCE",
        fill=(
            255,
            255,
            255,
        ),
    )

    filename = (
        f"{uuid.uuid4().hex}"
        "_optical_sar_fusion.jpg"
    )

    path = (
        output_dir
        / filename
    )

    evidence.save(
        path,
        format="JPEG",
        quality=88,
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    optical_mean = float(
        optical_intensity.mean()
    )

    sar_mean = float(
        sar_intensity.mean()
    )

    optical_std = float(
        optical_intensity.std()
    )

    sar_std = float(
        sar_intensity.std()
    )

    # ========================================================
    # CORRELATION
    # ========================================================

    correlation = None

    try:

        flat_o = (
            optical_intensity
            .reshape(-1)
        )

        flat_s = (
            sar_intensity
            .reshape(-1)
        )

        if (
            flat_o.std() > 1e-8
            and flat_s.std() > 1e-8
        ):

            correlation = float(
                np.corrcoef(
                    flat_o,
                    flat_s,
                )[0, 1]
            )

    except Exception:
        pass

    # ========================================================
    # GEOSPATIAL INFORMATION
    # ========================================================

    geospatial = {
        "optical": {
            "geospatial": optical_meta.get(
                "geospatial",
                False,
            ),

            "crs": optical_meta.get(
                "crs"
            ),

            "bounds": optical_meta.get(
                "bounds"
            ),

            "resolution": optical_meta.get(
                "resolution"
            ),
        },

        "sar": {
            "geospatial": sar_meta.get(
                "geospatial",
                False,
            ),

            "crs": sar_meta.get(
                "crs"
            ),

            "bounds": sar_meta.get(
                "bounds"
            ),

            "resolution": sar_meta.get(
                "resolution"
            ),
        },
    }

    return (
        f"/generated/{filename}",
        {
            "method": (
                "normalized optical intensity + "
                "normalized SAR intensity fusion"
            ),

            "optical_mean_intensity": round(
                optical_mean,
                4,
            ),

            "sar_mean_intensity": round(
                sar_mean,
                4,
            ),

            "optical_texture_std": round(
                optical_std,
                4,
            ),

            "sar_texture_std": round(
                sar_std,
                4,
            ),

            "optical_sar_correlation": (
                round(
                    correlation,
                    4,
                )
                if correlation is not None
                else None
            ),

            "geospatial": geospatial,

            "interpretation": (
                "Optical contributes spectral and "
                "contextual appearance, while SAR "
                "contributes complementary radar "
                "backscatter, texture and structural "
                "surface information."
            ),

            "warning": (
                "Optical-SAR fusion evidence is a "
                "visual/computational support layer "
                "and does not by itself constitute "
                "a calibrated physical sensor-fusion product."
            ),
        },
    )


# ============================================================
# GEOTIFF AREA UTILITY
# ============================================================

def calculate_geotiff_area(
    image_bytes: bytes,
    filename: str,
) -> dict:

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in {
        ".tif",
        ".tiff",
    }:

        return {
            "available": False,
            "reason": (
                "Input is not a TIFF/GeoTIFF."
            ),
        }

    try:

        import rasterio

        with rasterio.MemoryFile(
            image_bytes
        ) as memfile:

            with memfile.open() as src:

                geo = (
                    _calculate_geospatial_metrics(
                        src
                    )
                )

                if geo.get(
                    "scene_area"
                ) is None:

                    return {
                        "available": False,
                        "geospatial": geo,
                        "reason": (
                            "Scene area could not "
                            "be calculated from the CRS."
                        ),
                    }

                return {
                    "available": True,
                    "filename": filename,
                    "crs": geo.get(
                        "crs"
                    ),
                    "resolution": geo.get(
                        "resolution"
                    ),
                    "bounds": geo.get(
                        "bounds"
                    ),
                    "wgs84_bounds": geo.get(
                        "wgs84_bounds"
                    ),
                    "center_latitude_longitude": (
                        geo.get(
                            "center_latitude_longitude"
                        )
                    ),
                    "scene_area": geo.get(
                        "scene_area"
                    ),
                }

    except Exception as error:

        return {
            "available": False,
            "reason": (
                "Geospatial analysis failed: "
                f"{type(error).__name__}"
            ),
        }