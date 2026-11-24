from __future__ import annotations

import io
import json
import os
from html import escape
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from fastapi.responses import StreamingResponse

from pydantic import BaseModel, Field


# ============================================================
# INTERNAL MODULES
# ============================================================

from ai_model import analyze_satellite_images

from image_utils import validate_image


# ============================================================
# REMOTE SENSING TOOLS
# ============================================================

# These imports connect the new remote-sensing analysis
# functions to the API layer.
#
# If ai_model.py already calls these internally, the imports
# are still useful for direct API-level enrichment.

from rs_tools import (
    raster_analysis,
    create_change_map,
    create_optical_sar_evidence,
    calculate_geotiff_area,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GENERATED_DIR = BASE_DIR / "generated"

GENERATED_DIR.mkdir(
    exist_ok=True,
    parents=True,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SatQuery AI",
    description=(
        "Agentic Vision-Language Assistant "
        "for Multimodal Remote Sensing"
    ),
    version="3.2.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# STATIC GENERATED FILES
# ============================================================

app.mount(
    "/generated",

    StaticFiles(
        directory=str(GENERATED_DIR)
    ),

    name="generated",
)


# ============================================================
# REPORT REQUEST MODEL
# ============================================================

class ReportRequest(BaseModel):

    success: bool | None = None

    question: str | None = None

    analysis_mode: str | None = None

    modality: str | None = None

    images: list[dict[str, Any]] = Field(
        default_factory=list
    )

    task: str | None = None

    models: list[dict[str, Any]] = Field(
        default_factory=list
    )

    tools: list[str] = Field(
        default_factory=list
    )

    answer: str | None = None

    confidence: str | None = None

    evidence: dict[str, Any] = Field(
        default_factory=dict
    )

    execution_summary: dict[str, Any] = Field(
        default_factory=dict
    )

    adaptation: dict[str, Any] = Field(
        default_factory=dict
    )

    evidence_image: str | None = None


# ============================================================
# HEALTH / HOME
# ============================================================

@app.get("/")
def home():

    return {
        "success": True,

        "message": (
            "SatQuery AI Backend is Running 🚀"
        ),

        "version": "3.2.0",

        "capabilities": [
            "Remote sensing VQA",
            "Scene captioning",
            "Change detection",
            "Optical + SAR fusion",
            "GeoTIFF analysis",
            "NDVI analysis",
            "NDWI analysis",
            "NDBI analysis",
            "Band statistics",
            "SAR VV/VH analysis",
            "Latitude / longitude extraction",
            "CRS detection",
            "WGS84 bounds extraction",
            "Scene area calculation",
            "Changed area calculation",
            "Change component detection",
            "Spatial change distribution",
            "Visual evidence generation",
            "PDF report generation",
        ],
    }


@app.get("/health")
def health():

    return {
        "status": "online",

        "service": "SatQuery AI",

        "version": "3.2.0",

        "groq_configured": bool(
            os.getenv("GROQ_API_KEY")
        ),

        "remote_sensing_tools": True,

        "spectral_indices": {
            "NDVI": True,
            "NDWI": True,
            "NDBI": True,
        },

        "sar_analysis": True,

        "optical_sar_fusion": True,

        "visual_evidence": True,

        "geospatial_analysis": True,

        "latitude_longitude": True,

        "crs_detection": True,

        "scene_area": True,

        "change_detection": True,

        "report_generation": True,
    }


# ============================================================
# GEOLOCATION EXTRACTION
# ============================================================

def extract_location_from_analysis(
    analysis_result: dict[str, Any],
) -> dict[str, Any]:

    """
    Extract geographic information from computational
    remote-sensing analysis.

    Priority:

    1. center_latitude_longitude
    2. WGS84 bounds
    3. CRS
    4. projected bounds
    5. resolution
    6. scene area
    """

    location = {
        "available": False,

        "latitude": None,

        "longitude": None,

        "crs": None,

        "coordinate_system_type": None,

        "wgs84_bounds": None,

        "bounds": None,

        "resolution": None,

        "scene_area": None,
    }

    evidence = analysis_result.get(
        "evidence",
        {}
    )

    computational = evidence.get(
        "computational",
        []
    )

    # --------------------------------------------------------
    # Search computational evidence
    # --------------------------------------------------------

    for item in computational:

        if not isinstance(item, dict):
            continue

        raster_result = item.get(
            "raster_analysis",
            {}
        )

        if not raster_result:
            continue

        geospatial = raster_result.get(
            "geospatial",
            {}
        )

        if not isinstance(
            geospatial,
            dict
        ):
            continue

        center = geospatial.get(
            "center_latitude_longitude"
        )

        if center:

            location.update(
                {
                    "available": True,

                    "latitude": center.get(
                        "latitude"
                    ),

                    "longitude": center.get(
                        "longitude"
                    ),

                    "crs": geospatial.get(
                        "crs"
                    ),

                    "coordinate_system_type": (
                        geospatial.get(
                            "coordinate_system_type"
                        )
                    ),

                    "wgs84_bounds": (
                        geospatial.get(
                            "wgs84_bounds"
                        )
                    ),

                    "bounds": (
                        geospatial.get(
                            "bounds"
                        )
                    ),

                    "resolution": (
                        geospatial.get(
                            "resolution"
                        )
                    ),

                    "scene_area": (
                        geospatial.get(
                            "scene_area"
                        )
                    ),
                }
            )

            return location

        # ----------------------------------------------------
        # CRS fallback
        # ----------------------------------------------------

        if geospatial.get("crs"):

            location.update(
                {
                    "available": True,

                    "crs": geospatial.get(
                        "crs"
                    ),

                    "coordinate_system_type": (
                        geospatial.get(
                            "coordinate_system_type"
                        )
                    ),

                    "wgs84_bounds": (
                        geospatial.get(
                            "wgs84_bounds"
                        )
                    ),

                    "bounds": (
                        geospatial.get(
                            "bounds"
                        )
                    ),

                    "resolution": (
                        geospatial.get(
                            "resolution"
                        )
                    ),

                    "scene_area": (
                        geospatial.get(
                            "scene_area"
                        )
                    ),
                }
            )

            return location

    return location


# ============================================================
# IMAGE METADATA LOCATION EXTRACTION
# ============================================================

def extract_location_from_metadata(
    images: list[dict[str, Any]],
) -> dict[str, Any]:

    """
    Fallback location extraction directly from uploaded
    image metadata.

    Note:
    If bounds are in a projected CRS, the center values are
    projected coordinates rather than latitude/longitude.
    Therefore this is only a fallback.
    """

    location = {
        "available": False,

        "latitude": None,

        "longitude": None,

        "crs": None,

        "bounds": None,

        "resolution": None,

        "scene_area": None,
    }

    for image in images:

        metadata = image.get(
            "metadata",
            {}
        )

        if not metadata.get(
            "geospatial",
            False
        ):
            continue

        bounds = metadata.get(
            "bounds"
        )

        crs = metadata.get(
            "crs"
        )

        resolution = metadata.get(
            "resolution"
        )

        if bounds and len(bounds) == 4:

            try:

                left = float(bounds[0])

                bottom = float(bounds[1])

                right = float(bounds[2])

                top = float(bounds[3])

                center_x = (
                    left + right
                ) / 2

                center_y = (
                    bottom + top
                ) / 2

                location.update(
                    {
                        "available": True,

                        "latitude": (
                            center_y
                            if not crs
                            else None
                        ),

                        "longitude": (
                            center_x
                            if not crs
                            else None
                        ),

                        "crs": crs,

                        "bounds": [
                            left,
                            bottom,
                            right,
                            top,
                        ],

                        "resolution": resolution,
                    }
                )

                return location

            except Exception:
                pass

    return location


# ============================================================
# REMOTE-SENSING COMPUTATIONAL ENRICHMENT
# ============================================================

def run_remote_sensing_tools(
    image_data: list[dict[str, Any]],
    analysis_mode: str,
) -> dict[str, Any]:

    """
    Run the computational remote-sensing tools independently
    of the language model.

    This guarantees that NDVI / NDWI / NDBI / CRS / area /
    SAR / change-map information is available to the API
    even when the language-model layer does not explicitly
    expose every field.
    """

    computational_results = []

    # ========================================================
    # SINGLE IMAGE ANALYSIS
    # ========================================================

    if len(image_data) >= 1:

        first = image_data[0]

        try:

            raster_result = raster_analysis(
                first["bytes"],
                first["filename"],
            )

            computational_results.append(
                {
                    "filename": first["filename"],

                    "raster_analysis": raster_result,
                }
            )

        except Exception as error:

            computational_results.append(
                {
                    "filename": first["filename"],

                    "raster_analysis_error": (
                        f"{type(error).__name__}: "
                        f"{str(error)}"
                    ),
                }
            )

    # ========================================================
    # SECOND IMAGE
    # ========================================================

    if len(image_data) == 2:

        second = image_data[1]

        try:

            raster_result = raster_analysis(
                second["bytes"],
                second["filename"],
            )

            computational_results.append(
                {
                    "filename": second["filename"],

                    "raster_analysis": raster_result,
                }
            )

        except Exception as error:

            computational_results.append(
                {
                    "filename": second["filename"],

                    "raster_analysis_error": (
                        f"{type(error).__name__}: "
                        f"{str(error)}"
                    ),
                }
            )

    return {
        "computational": computational_results
    }


# ============================================================
# EXTRACT SPECTRAL INDICES
# ============================================================

def extract_spectral_results(
    computational: list[dict[str, Any]],
) -> dict[str, Any]:

    """
    Collect NDVI / NDWI / NDBI and semantic indicators from
    all raster analyses.
    """

    result = {
        "NDVI": None,
        "NDVI_mean": None,
        "NDVI_median": None,

        "NDWI": None,
        "NDWI_mean": None,
        "NDWI_median": None,

        "NDBI": None,
        "NDBI_mean": None,
        "NDBI_median": None,

        "NDVI_statistics": None,
        "NDWI_statistics": None,
        "NDBI_statistics": None,

        "semantic_indicators": [],

        "per_image": [],
    }

    for item in computational:

        raster = item.get(
            "raster_analysis"
        )

        if not raster:
            continue

        indices = raster.get(
            "indices",
            {}
        )

        index_statistics = raster.get(
            "index_statistics",
            {}
        )

        semantic = raster.get(
            "semantic_indicators",
            {}
        )

        image_result = {
            "filename": item.get(
                "filename"
            ),

            "indices": indices,

            "index_statistics": (
                index_statistics
            ),

            "semantic_indicators": (
                semantic
            ),
        }

        result["per_image"].append(
            image_result
        )

        # ----------------------------------------------------
        # NDVI
        # ----------------------------------------------------

        if result["NDVI"] is None:

            result["NDVI"] = indices.get(
                "NDVI"
            )

            result["NDVI_mean"] = indices.get(
                "NDVI_mean"
            )

            result["NDVI_median"] = indices.get(
                "NDVI_median"
            )

            result["NDVI_statistics"] = (
                index_statistics.get(
                    "NDVI"
                )
            )

        # ----------------------------------------------------
        # NDWI
        # ----------------------------------------------------

        if result["NDWI"] is None:

            result["NDWI"] = indices.get(
                "NDWI"
            )

            result["NDWI_mean"] = indices.get(
                "NDWI_mean"
            )

            result["NDWI_median"] = indices.get(
                "NDWI_median"
            )

            result["NDWI_statistics"] = (
                index_statistics.get(
                    "NDWI"
                )
            )

        # ----------------------------------------------------
        # NDBI
        # ----------------------------------------------------

        if result["NDBI"] is None:

            result["NDBI"] = indices.get(
                "NDBI"
            )

            result["NDBI_mean"] = indices.get(
                "NDBI_mean"
            )

            result["NDBI_median"] = indices.get(
                "NDBI_median"
            )

            result["NDBI_statistics"] = (
                index_statistics.get(
                    "NDBI"
                )
            )

        # ----------------------------------------------------
        # Semantic indicators
        # ----------------------------------------------------

        if semantic:

            result[
                "semantic_indicators"
            ].append(
                {
                    "filename": item.get(
                        "filename"
                    ),

                    **semantic,
                }
            )

    return result


# ============================================================
# EXTRACT SAR RESULTS
# ============================================================

def extract_sar_results(
    computational: list[dict[str, Any]],
) -> dict[str, Any]:

    result = {
        "available": False,

        "per_image": [],

        "vv_vh_ratio_mean": None,
    }

    for item in computational:

        raster = item.get(
            "raster_analysis"
        )

        if not raster:
            continue

        sar = raster.get(
            "sar_analysis"
        )

        if not sar:
            continue

        result["available"] = True

        result["per_image"].append(
            {
                "filename": item.get(
                    "filename"
                ),

                **sar,
            }
        )

        if (
            result[
                "vv_vh_ratio_mean"
            ] is None
        ):

            result[
                "vv_vh_ratio_mean"
            ] = sar.get(
                "vv_vh_ratio_mean"
            )

    return result


# ============================================================
# EXTRACT GEOSPATIAL RESULTS
# ============================================================

def extract_geospatial_results(
    computational: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    result = []

    for item in computational:

        raster = item.get(
            "raster_analysis"
        )

        if not raster:
            continue

        geo = raster.get(
            "geospatial",
            {}
        )

        if not geo:
            continue

        result.append(
            {
                "filename": item.get(
                    "filename"
                ),

                **geo,
            }
        )

    return result


# ============================================================
# CHANGE DETECTION
# ============================================================

def run_change_detection(
    image_data: list[dict[str, Any]],
    analysis_mode: str,
) -> dict[str, Any]:

    """
    Create computational change-map evidence for two temporal
    images.
    """

    if len(image_data) != 2:

        return {
            "available": False,

            "reason": (
                "Change detection requires "
                "exactly two images."
            ),
        }

    try:

        change_map_url, change_metrics = (
            create_change_map(

                image_data[0]["bytes"],

                image_data[0]["filename"],

                image_data[1]["bytes"],

                image_data[1]["filename"],

                GENERATED_DIR,
            )
        )

        return {
            "available": True,

            "before": image_data[0]["filename"],

            "after": image_data[1]["filename"],

            "change_map": change_map_url,

            "metrics": change_metrics,
        }

    except Exception as error:

        return {
            "available": False,

            "reason": (
                f"Change detection failed: "
                f"{type(error).__name__}: "
                f"{str(error)}"
            ),
        }


# ============================================================
# OPTICAL + SAR FUSION
# ============================================================

def run_optical_sar_fusion(
    image_data: list[dict[str, Any]],
    analysis_mode: str,
) -> dict[str, Any]:

    """
    Generate Optical + SAR computational evidence.
    """

    if len(image_data) != 2:

        return {
            "available": False,

            "reason": (
                "Optical + SAR fusion requires "
                "exactly two images."
            ),
        }

    if analysis_mode != "optical-sar":

        return {
            "available": False,

            "reason": (
                "Optical + SAR fusion is only "
                "enabled for optical-sar mode."
            ),
        }

    try:

        fusion_url, fusion_metrics = (
            create_optical_sar_evidence(

                image_data[0]["bytes"],

                image_data[0]["filename"],

                image_data[1]["bytes"],

                image_data[1]["filename"],

                GENERATED_DIR,
            )
        )

        return {
            "available": True,

            "optical": image_data[0][
                "filename"
            ],

            "sar": image_data[1][
                "filename"
            ],

            "fusion_image": fusion_url,

            "metrics": fusion_metrics,
        }

    except Exception as error:

        return {
            "available": False,

            "reason": (
                f"Optical-SAR fusion failed: "
                f"{type(error).__name__}: "
                f"{str(error)}"
            ),
        }


# ============================================================
# COMBINE GENERATED EVIDENCE
# ============================================================

def select_primary_evidence_image(
    result: dict[str, Any],
    change_result: dict[str, Any],
    fusion_result: dict[str, Any],
) -> str | None:

    """
    Select the most relevant generated visual for the
    frontend's primary evidence image.
    """

    # Change map gets highest priority.
    if change_result.get(
        "available"
    ):

        return change_result.get(
            "change_map"
        )

    # Optical + SAR fusion next.
    if fusion_result.get(
        "available"
    ):

        return fusion_result.get(
            "fusion_image"
        )

    # Otherwise use the AI-model evidence image.
    return result.get(
        "evidence_image"
    )


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze_image(

    files: list[UploadFile] = File(...),

    question: str = Form(...),

    analysis_mode: str = Form("auto"),

    modality: str = Form("auto"),
):

    # ========================================================
    # QUESTION VALIDATION
    # ========================================================

    if not question.strip():

        raise HTTPException(
            status_code=400,

            detail=(
                "Please enter a question "
                "about the satellite image."
            ),
        )

    # ========================================================
    # IMAGE COUNT VALIDATION
    # ========================================================

    if len(files) < 1 or len(files) > 2:

        raise HTTPException(
            status_code=400,

            detail=(
                "Please upload one or two "
                "satellite images."
            ),
        )

    # ========================================================
    # CHANGE MODE
    # ========================================================

    if (
        analysis_mode == "change"
        and len(files) != 2
    ):

        raise HTTPException(
            status_code=400,

            detail=(
                "Bi-temporal analysis requires "
                "exactly two images."
            ),
        )

    # ========================================================
    # OPTICAL + SAR MODE
    # ========================================================

    if (
        analysis_mode == "optical-sar"
        and len(files) != 2
    ):

        raise HTTPException(
            status_code=400,

            detail=(
                "Optical + SAR analysis requires "
                "exactly two images."
            ),
        )

    # ========================================================
    # READ + VALIDATE IMAGES
    # ========================================================

    image_data = []

    for uploaded_file in files:

        try:

            contents = await uploaded_file.read()

            if not contents:

                raise ValueError(
                    "Empty file."
                )

            filename = (
                uploaded_file.filename
                or "uploaded_image"
            )

            metadata = validate_image(
                contents,

                filename,

                uploaded_file.content_type,
            )

            image_data.append(
                {
                    "filename": filename,

                    "content_type": (
                        uploaded_file.content_type
                    ),

                    "bytes": contents,

                    "metadata": metadata,
                }
            )

        except Exception as error:

            raise HTTPException(
                status_code=400,

                detail=(
                    f"Invalid image "
                    f"'{uploaded_file.filename}': "
                    f"{str(error)}"
                ),
            )

    # ========================================================
    # AI ANALYSIS
    # ========================================================

    try:

        result = analyze_satellite_images(

            images=image_data,

            question=question,

            analysis_mode=analysis_mode,

            modality=modality,

            generated_dir=GENERATED_DIR,
        )

    except Exception as error:

        print(
            "SATQUERY ANALYSIS ERROR:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,

            detail=(
                f"AI analysis failed: "
                f"{str(error)}"
            ),
        )

    # ========================================================
    # COMPUTATIONAL REMOTE-SENSING ANALYSIS
    # ========================================================

    try:

        computational_wrapper = (
            run_remote_sensing_tools(
                image_data,
                analysis_mode,
            )
        )

    except Exception as error:

        print(
            "REMOTE SENSING TOOL ERROR:",
            repr(error),
        )

        computational_wrapper = {
            "computational": []
        }

    computational = (
        computational_wrapper.get(
            "computational",
            []
        )
    )

    # ========================================================
    # EXISTING AI EVIDENCE
    # ========================================================

    existing_evidence = result.get(
        "evidence",
        {}
    )

    if not isinstance(
        existing_evidence,
        dict
    ):

        existing_evidence = {}

    # ========================================================
    # MERGE COMPUTATIONAL EVIDENCE
    # ========================================================

    existing_computational = (
        existing_evidence.get(
            "computational",
            []
        )
    )

    if not isinstance(
        existing_computational,
        list
    ):

        existing_computational = []

    # Use the direct computational analysis as the reliable
    # API source.
    merged_computational = (
        existing_computational
        + computational
    )

    # Avoid duplicate exact object entries.
    unique_computational = []

    seen = set()

    for item in merged_computational:

        try:

            key = json.dumps(
                item,
                sort_keys=True,
                default=str,
            )

        except Exception:

            key = str(item)

        if key in seen:
            continue

        seen.add(key)

        unique_computational.append(
            item
        )

    existing_evidence[
        "computational"
    ] = unique_computational

    # ========================================================
    # SPECTRAL INDICES
    # ========================================================

    spectral_results = (
        extract_spectral_results(
            computational
        )
    )

    # ========================================================
    # SAR RESULTS
    # ========================================================

    sar_results = (
        extract_sar_results(
            computational
        )
    )

    # ========================================================
    # GEOSPATIAL RESULTS
    # ========================================================

    geospatial_results = (
        extract_geospatial_results(
            computational
        )
    )

    # ========================================================
    # CHANGE DETECTION
    # ========================================================

    change_result = {
        "available": False
    }

    if (
        analysis_mode == "change"
        or (
            len(image_data) == 2
            and analysis_mode == "auto"
        )
    ):

        change_result = (
            run_change_detection(
                image_data,
                analysis_mode,
            )
        )

    # ========================================================
    # OPTICAL + SAR
    # ========================================================

    fusion_result = (
        run_optical_sar_fusion(
            image_data,
            analysis_mode,
        )
    )

    # ========================================================
    # ADD CHANGE EVIDENCE
    # ========================================================

    if change_result.get(
        "available"
    ):

        existing_evidence[
            "change_detection"
        ] = change_result

    # ========================================================
    # ADD SAR EVIDENCE
    # ========================================================

    if sar_results.get(
        "available"
    ):

        existing_evidence[
            "sar_analysis"
        ] = sar_results

    # ========================================================
    # ADD OPTICAL + SAR EVIDENCE
    # ========================================================

    if fusion_result.get(
        "available"
    ):

        existing_evidence[
            "optical_sar_fusion"
        ] = fusion_result

    # ========================================================
    # ADD SPECTRAL EVIDENCE
    # ========================================================

    existing_evidence[
        "spectral_indices"
    ] = spectral_results

    # ========================================================
    # ADD GEOSPATIAL EVIDENCE
    # ========================================================

    existing_evidence[
        "geospatial_results"
    ] = geospatial_results

    # ========================================================
    # EXTRACT LOCATION
    # ========================================================

    location = (
        extract_location_from_analysis(
            {
                "evidence": existing_evidence
            }
        )
    )

    # --------------------------------------------------------
    # Fallback to uploaded metadata
    # --------------------------------------------------------

    if not location.get(
        "available"
    ):

        metadata_location = (
            extract_location_from_metadata(
                image_data
            )
        )

        if metadata_location.get(
            "available"
        ):

            location.update(
                metadata_location
            )

    # ========================================================
    # BUILD GEOSPATIAL SUMMARY
    # ========================================================

    geospatial_summary = {
        "available": bool(
            geospatial_results
        ),

        "images": geospatial_results,

        "primary": (
            geospatial_results[0]
            if geospatial_results
            else {}
        ),
    }

    # ========================================================
    # EXECUTION SUMMARY
    # ========================================================

    execution_summary = result.get(
        "execution_summary",
        {}
    )

    if not isinstance(
        execution_summary,
        dict
    ):

        execution_summary = {}

    execution_summary[
        "geospatial_analysis"
    ] = (
        "available"
        if location.get(
            "available"
        )
        else "not available"
    )

    execution_summary[
        "latitude_longitude_extracted"
    ] = bool(
        location.get(
            "latitude"
        ) is not None
        and
        location.get(
            "longitude"
        ) is not None
    )

    execution_summary[
        "crs_detected"
    ] = bool(
        location.get(
            "crs"
        )
    )

    execution_summary[
        "scene_area_available"
    ] = bool(
        location.get(
            "scene_area"
        )
    )

    execution_summary[
        "spectral_indices_available"
    ] = bool(
        spectral_results.get(
            "NDVI"
        ) is not None
        or
        spectral_results.get(
            "NDWI"
        ) is not None
        or
        spectral_results.get(
            "NDBI"
        ) is not None
    )

    execution_summary[
        "sar_analysis_available"
    ] = bool(
        sar_results.get(
            "available"
        )
    )

    execution_summary[
        "change_detection_available"
    ] = bool(
        change_result.get(
            "available"
        )
    )

    execution_summary[
        "optical_sar_fusion_available"
    ] = bool(
        fusion_result.get(
            "available"
        )
    )

    # ========================================================
    # PRIMARY EVIDENCE IMAGE
    # ========================================================

    evidence_image = (
        select_primary_evidence_image(
            result,

            change_result,

            fusion_result,
        )
    )

    # ========================================================
    # FINAL API RESPONSE
    # ========================================================

    return {

        "success": True,

        "question": question,

        "analysis_mode": analysis_mode,

        "modality": modality,

        # ====================================================
        # INPUT IMAGES
        # ====================================================

        "images": [

            {
                "filename": item[
                    "filename"
                ],

                "content_type": item.get(
                    "content_type"
                ),

                "metadata": item[
                    "metadata"
                ],
            }

            for item in image_data
        ],

        # ====================================================
        # AI ANALYSIS
        # ====================================================

        "task": result.get(
            "task"
        ),

        "models": result.get(
            "models",
            []
        ),

        "tools": result.get(
            "tools",
            []
        ),

        "answer": result.get(
            "answer"
        ),

        "confidence": result.get(
            "confidence"
        ),

        # ====================================================
        # LOCATION
        # ====================================================

        "location": location,

        # ====================================================
        # GEOSPATIAL
        # ====================================================

        "geospatial": geospatial_summary,

        # ====================================================
        # SPECTRAL INDICES
        # ====================================================

        "spectral_indices": spectral_results,

        # Easy frontend access
        "indices": {

            "NDVI": spectral_results.get(
                "NDVI"
            ),

            "NDVI_mean": spectral_results.get(
                "NDVI_mean"
            ),

            "NDVI_median": spectral_results.get(
                "NDVI_median"
            ),

            "NDWI": spectral_results.get(
                "NDWI"
            ),

            "NDWI_mean": spectral_results.get(
                "NDWI_mean"
            ),

            "NDWI_median": spectral_results.get(
                "NDWI_median"
            ),

            "NDBI": spectral_results.get(
                "NDBI"
            ),

            "NDBI_mean": spectral_results.get(
                "NDBI_mean"
            ),

            "NDBI_median": spectral_results.get(
                "NDBI_median"
            ),
        },

        # ====================================================
        # SAR
        # ====================================================

        "sar": sar_results,

        # ====================================================
        # CHANGE DETECTION
        # ====================================================

        "change_detection": change_result,

        # ====================================================
        # OPTICAL + SAR
        # ====================================================

        "optical_sar": fusion_result,

        # ====================================================
        # EVIDENCE
        # ====================================================

        "evidence": existing_evidence,

        # ====================================================
        # EXECUTION
        # ====================================================

        "execution_summary": execution_summary,

        # ====================================================
        # DOMAIN ADAPTATION
        # ====================================================

        "adaptation": result.get(
            "adaptation",
            {}
        ),

        # ====================================================
        # GENERATED VISUAL EVIDENCE
        # ====================================================

        "evidence_image": evidence_image,
    }


# ============================================================
# REPORT HELPERS
# ============================================================

def safe_html(
    value: Any
) -> str:

    """
    Safely convert arbitrary values into HTML-safe text
    for ReportLab Paragraph objects.
    """

    if value is None:
        return ""

    return escape(
        str(value)
    ).replace(
        "\n",
        "<br/>"
    )


def pretty_json(
    value: Any
) -> str:

    try:

        text = json.dumps(
            value,

            indent=2,

            ensure_ascii=False,

            default=str,
        )

        return safe_html(
            text
        )

    except Exception:

        return safe_html(
            str(value)
        )


# ============================================================
# REPORT ENDPOINT
# ============================================================

@app.post("/report")
def create_report(
    payload: ReportRequest
):

    # ========================================================
    # REPORTLAB IMPORT
    # ========================================================

    try:

        from reportlab.lib.pagesizes import A4

        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle,
        )

        from reportlab.lib.enums import (
            TA_CENTER,
        )

        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image as ReportImage,
        )

        from reportlab.lib import colors

    except ImportError:

        raise HTTPException(
            status_code=500,

            detail=(
                "ReportLab is not installed. "
                "Run: pip install reportlab"
            ),
        )

    # ========================================================
    # BUFFER
    # ========================================================

    buffer = io.BytesIO()

    # ========================================================
    # DOCUMENT
    # ========================================================

    doc = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=40,

        leftMargin=40,

        topMargin=40,

        bottomMargin=40,
    )

    # ========================================================
    # STYLES
    # ========================================================

    styles = getSampleStyleSheet()

    title = ParagraphStyle(

        "SatQueryTitle",

        parent=styles["Title"],

        fontSize=22,

        leading=26,

        alignment=TA_CENTER,

        spaceAfter=14,
    )

    subtitle = ParagraphStyle(

        "SatQuerySubtitle",

        parent=styles["BodyText"],

        fontSize=10,

        leading=14,

        alignment=TA_CENTER,

        spaceAfter=16,
    )

    heading = ParagraphStyle(

        "SatQueryHeading",

        parent=styles["Heading2"],

        fontSize=13,

        leading=16,

        spaceBefore=12,

        spaceAfter=6,
    )

    body = ParagraphStyle(

        "SatQueryBody",

        parent=styles["BodyText"],

        fontSize=9,

        leading=13,

        spaceAfter=6,
    )

    small = ParagraphStyle(

        "SatQuerySmall",

        parent=styles["BodyText"],

        fontSize=7,

        leading=10,

        spaceAfter=4,
    )

    # ========================================================
    # LOCATION
    # ========================================================

    location = (
        extract_location_from_analysis(
            {
                "evidence": payload.evidence
            }
        )
    )

    if payload.evidence.get(
        "location"
    ):

        location.update(
            payload.evidence.get(
                "location"
            )
        )

    latitude = location.get(
        "latitude"
    )

    longitude = location.get(
        "longitude"
    )

    crs = location.get(
        "crs"
    )

    # ========================================================
    # STORY
    # ========================================================

    story = []

    story.append(
        Paragraph(
            "SatQuery AI — Analysis Report",

            title,
        )
    )

    story.append(
        Paragraph(
            "Agentic Remote-Sensing Intelligence",

            subtitle,
        )
    )

    story.append(
        Spacer(
            1,
            8,
        )
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "Analysis",
            heading,
        )
    )

    story.append(
        Paragraph(
            (
                f"<b>Question:</b> "
                f"{safe_html(payload.question)}"
            ),
            body,
        )
    )

    story.append(
        Paragraph(
            (
                f"<b>Task:</b> "
                f"{safe_html(payload.task)}"
            ),
            body,
        )
    )

    story.append(
        Paragraph(
            (
                f"<b>Mode:</b> "
                f"{safe_html(payload.analysis_mode)}"
            ),
            body,
        )
    )

    story.append(
        Paragraph(
            (
                f"<b>Modality:</b> "
                f"{safe_html(payload.modality)}"
            ),
            body,
        )
    )

    story.append(
        Paragraph(
            (
                f"<b>Confidence:</b> "
                f"{safe_html(payload.confidence or 'Not available')}"
            ),
            body,
        )
    )

    # ========================================================
    # AI ANSWER
    # ========================================================

    story.append(
        Paragraph(
            "AI Answer",
            heading,
        )
    )

    story.append(
        Paragraph(
            safe_html(
                payload.answer
                or "No answer available."
            ),
            body,
        )
    )

    # ========================================================
    # SPECTRAL INDICES
    # ========================================================

    evidence = payload.evidence or {}

    spectral = evidence.get(
        "spectral_indices",
        {}
    )

    story.append(
        Paragraph(
            "Spectral Indices",
            heading,
        )
    )

    indices_table = [
        [
            "Index",
            "Value",
            "Meaning",
        ]
    ]

    ndvi = spectral.get(
        "NDVI"
    )

    ndwi = spectral.get(
        "NDWI"
    )

    ndbi = spectral.get(
        "NDBI"
    )

    indices_table.append(
        [
            "NDVI",
            str(
                ndvi
                if ndvi is not None
                else "N/A"
            ),
            "Vegetation indicator",
        ]
    )

    indices_table.append(
        [
            "NDWI",
            str(
                ndwi
                if ndwi is not None
                else "N/A"
            ),
            "Water/moisture indicator",
        ]
    )

    indices_table.append(
        [
            "NDBI",
            str(
                ndbi
                if ndbi is not None
                else "N/A"
            ),
            "Built-up indicator",
        ]
    )

    index_table = Table(
        indices_table,
        repeatRows=1,
    )

    index_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#182234"
                    ),
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
            ]
        )
    )

    story.append(
        index_table
    )

    # ========================================================
    # GEOSPATIAL LOCATION
    # ========================================================

    story.append(
        Paragraph(
            "Geospatial Information",
            heading,
        )
    )

    if (
        latitude is not None
        and longitude is not None
    ):

        story.append(
            Paragraph(
                (
                    f"<b>Latitude:</b> "
                    f"{latitude}°"
                ),
                body,
            )
        )

        story.append(
            Paragraph(
                (
                    f"<b>Longitude:</b> "
                    f"{longitude}°"
                ),
                body,
            )
        )

    else:

        story.append(
            Paragraph(
                (
                    "Latitude / longitude: "
                    "Not available for this input."
                ),
                body,
            )
        )

    if crs:

        story.append(
            Paragraph(
                (
                    f"<b>CRS:</b> "
                    f"{safe_html(crs)}"
                ),
                body,
            )
        )

    coordinate_type = location.get(
        "coordinate_system_type"
    )

    if coordinate_type:

        story.append(
            Paragraph(
                (
                    "<b>Coordinate system:</b> "
                    f"{safe_html(coordinate_type)}"
                ),
                body,
            )
        )

    resolution = location.get(
        "resolution"
    )

    if resolution:

        story.append(
            Paragraph(
                (
                    "<b>Pixel resolution:</b> "
                    f"{safe_html(resolution)}"
                ),
                body,
            )
        )

    # ========================================================
    # SCENE AREA
    # ========================================================

    scene_area = location.get(
        "scene_area"
    )

    if scene_area:

        if scene_area.get(
            "square_kilometers"
        ) is not None:

            story.append(
                Paragraph(
                    (
                        "<b>Scene area:</b> "
                        f"{scene_area.get('square_kilometers')} km²"
                    ),
                    body,
                )
            )

        if scene_area.get(
            "square_meters"
        ) is not None:

            story.append(
                Paragraph(
                    (
                        "<b>Scene area:</b> "
                        f"{scene_area.get('square_meters')} m²"
                    ),
                    body,
                )
            )

    # ========================================================
    # WGS84 BOUNDS
    # ========================================================

    wgs84_bounds = location.get(
        "wgs84_bounds"
    )

    if wgs84_bounds:

        story.append(
            Paragraph(
                (
                    "<b>WGS84 bounds:</b> "
                    f"{safe_html(wgs84_bounds)}"
                ),
                small,
            )
        )

    # ========================================================
    # CHANGE DETECTION
    # ========================================================

    change = evidence.get(
        "change_detection"
    )

    if change:

        story.append(
            Paragraph(
                "Change Detection",
                heading,
            )
        )

        metrics = change.get(
            "metrics",
            {}
        )

        if metrics:

            story.append(
                Paragraph(
                    (
                        "<b>Changed area:</b> "
                        f"{metrics.get('changed_area_percent', 'N/A')}%"
                    ),
                    body,
                )
            )

            geo_change = metrics.get(
                "geospatial_change",
                {}
            )

            if geo_change.get(
                "available"
            ):

                story.append(
                    Paragraph(
                        (
                            "<b>Georeferenced changed area:</b> "
                            f"{geo_change.get('changed_area_km2', 'N/A')} km²"
                        ),
                        body,
                    )
                )

                story.append(
                    Paragraph(
                        (
                            "<b>Changed area:</b> "
                            f"{geo_change.get('changed_area_hectares', 'N/A')} hectares"
                        ),
                        body,
                    )
                )

        story.append(
            Paragraph(
                pretty_json(
                    change
                ),
                small,
            )
        )

    # ========================================================
    # SAR
    # ========================================================

    sar = evidence.get(
        "sar_analysis"
    )

    if sar:

        story.append(
            Paragraph(
                "SAR Analysis",
                heading,
            )
        )

        story.append(
            Paragraph(
                pretty_json(
                    sar
                ),
                small,
            )
        )

    # ========================================================
    # OPTICAL + SAR
    # ========================================================

    optical_sar = evidence.get(
        "optical_sar_fusion"
    )

    if optical_sar:

        story.append(
            Paragraph(
                "Optical + SAR Fusion",
                heading,
            )
        )

        story.append(
            Paragraph(
                pretty_json(
                    optical_sar
                ),
                small,
            )
        )

    # ========================================================
    # REMOTE SENSING ADAPTATION
    # ========================================================

    story.append(
        Paragraph(
            "Remote-Sensing Adaptation",
            heading,
        )
    )

    story.append(
        Paragraph(
            pretty_json(
                payload.adaptation
            ),
            small,
        )
    )

    # ========================================================
    # EXECUTION SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "Execution Summary",
            heading,
        )
    )

    story.append(
        Paragraph(
            pretty_json(
                payload.execution_summary
            ),
            small,
        )
    )

    # ========================================================
    # MODELS + TOOLS
    # ========================================================

    story.append(
        Paragraph(
            "Models and Tools",
            heading,
        )
    )

    for model in payload.models:

        story.append(
            Paragraph(
                (
                    f"<b>Model:</b> "
                    f"{safe_html(model.get('name', ''))}"
                    " — "
                    f"{safe_html(model.get('role', ''))}"
                ),
                body,
            )
        )

    story.append(
        Paragraph(
            "<b>Tools:</b> "
            +
            safe_html(
                ", ".join(
                    payload.tools
                )
            ),
            body,
        )
    )

    # ========================================================
    # INPUT METADATA
    # ========================================================

    story.append(
        Paragraph(
            "Input Metadata",
            heading,
        )
    )

    table_data = [
        [
            "Filename",
            "Width",
            "Height",
            "Format",
            "CRS",
            "Geo",
        ]
    ]

    for item in payload.images:

        metadata = item.get(
            "metadata",
            {}
        )

        table_data.append(
            [
                str(
                    item.get(
                        "filename",
                        ""
                    )
                ),

                str(
                    metadata.get(
                        "width",
                        ""
                    )
                ),

                str(
                    metadata.get(
                        "height",
                        ""
                    )
                ),

                str(
                    metadata.get(
                        "format",
                        ""
                    )
                ),

                str(
                    metadata.get(
                        "crs",
                        ""
                    )
                ),

                (
                    "Yes"
                    if metadata.get(
                        "geospatial",
                        False
                    )
                    else "No"
                ),
            ]
        )

    table = Table(
        table_data,
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#182234"
                    ),
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.extend(
        [
            table,

            Spacer(
                1,
                12,
            ),
        ]
    )

    # ========================================================
    # VISUAL EVIDENCE
    # ========================================================

    if payload.evidence_image:

        evidence_filename = (
            Path(
                payload.evidence_image
            ).name
        )

        evidence_path = (
            GENERATED_DIR
            / evidence_filename
        )

        if evidence_path.exists():

            story.append(
                Paragraph(
                    "Visual Evidence",
                    heading,
                )
            )

            try:

                report_image = ReportImage(

                    str(
                        evidence_path
                    ),

                    width=480,

                    height=260,

                    preserveAspectRatio=True,

                    anchor="c",
                )

                story.append(
                    report_image
                )

            except Exception as error:

                story.append(
                    Paragraph(
                        (
                            "Visual evidence could not "
                            "be embedded: "
                            f"{safe_html(error)}"
                        ),
                        small,
                    )
                )

    # ========================================================
    # EVIDENCE
    # ========================================================

    story.append(
        Paragraph(
            "Remote-Sensing Evidence",
            heading,
        )
    )

    story.append(
        Paragraph(
            pretty_json(
                payload.evidence
            ),
            small,
        )
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    try:

        doc.build(
            story
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,

            detail=(
                "Failed to generate PDF report: "
                f"{str(error)}"
            ),
        )

    # ========================================================
    # RETURN PDF
    # ========================================================

    buffer.seek(0)

    return StreamingResponse(

        buffer,

        media_type="application/pdf",

        headers={
            "Content-Disposition": (
                'attachment; '
                'filename="SatQuery_AI_Analysis_Report.pdf"'
            )
        },
    )


# ============================================================
# OPTIONAL DIRECT GEOTIFF AREA ENDPOINT
# ============================================================

@app.post("/geospatial/area")
async def geotiff_area(

    file: UploadFile = File(...),
):

    """
    Direct endpoint for calculating GeoTIFF scene area.

    Useful for frontend testing and future geospatial panels.
    """

    try:

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,

                detail="Empty file.",
            )

        filename = (
            file.filename
            or "uploaded.tif"
        )

        result = calculate_geotiff_area(
            contents,
            filename,
        )

        return {
            "success": True,

            **result,
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,

            detail=(
                f"Geospatial area analysis failed: "
                f"{str(error)}"
            ),
        )