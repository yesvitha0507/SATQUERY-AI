from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from rs_adapter import adapter
from rs_tools import (
    create_change_map,
    create_optical_sar_evidence,
    create_preview,
    raster_analysis,
)

# Pillow is used only to make the vision payload smaller.
from PIL import Image


# ============================================================
# ENVIRONMENT / GROQ
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Create a .env file and add GROQ_API_KEY."
    )

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {
    "single_vqa": {
        "model": "qwen/qwen3.6-27b",
        "description": "Remote-sensing visual question answering",
        "tools": [
            "input_validation",
            "remote_sensing_domain_adapter",
            "sentinel2_visual_encoder",
            "vision_vqa",
        ],
    },

    "captioning": {
        "model": "qwen/qwen3.6-27b",
        "description": "Remote-sensing scene description",
        "tools": [
            "input_validation",
            "remote_sensing_domain_adapter",
            "sentinel2_visual_encoder",
            "scene_captioning",
        ],
    },

    "change_analysis": {
        "model": "qwen/qwen3.6-27b",
        "description": "Bi-temporal change understanding",
        "tools": [
            "input_validation",
            "remote_sensing_domain_adapter",
            "sentinel2_visual_encoder",
            "image_registration_check",
            "change_detection_map",
            "change_vqa",
        ],
    },

    "optical_sar": {
        "model": "qwen/qwen3.6-27b",
        "description": "Optical-SAR joint information extraction",
        "tools": [
            "input_validation",
            "remote_sensing_domain_adapter",
            "sentinel2_visual_encoder",
            "optical_sar_fusion",
            "cross_modal_vqa",
        ],
    },
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are SatQuery AI, an evidence-grounded remote-sensing
vision-language assistant.

You analyze optical, multispectral, SAR and paired
remote-sensing imagery using visual, computational and
geospatial evidence.

Your final answer must look like a professional remote-sensing
analysis report, NOT like a casual chatbot response.

============================================================
CORE ANALYSIS RULES
============================================================

1. Optical imagery provides spectral and contextual evidence.

2. SAR provides radar backscatter, texture and structural
   evidence.

3. Never interpret SAR grayscale as normal optical RGB colour.

4. For bi-temporal imagery, compare spatially corresponding
   regions.

5. Use supplied computational evidence whenever available.

6. Never invent coordinates, locations, sensor metadata,
   spectral values or physical measurements.

7. Clearly distinguish direct observation from interpretation.

8. If evidence is insufficient, explicitly state that.

9. Do not reveal hidden reasoning or chain-of-thought.

10. Geospatial coordinates must come only from supplied
    metadata.

11. Do not claim a specific city, country or geographic place
    from visual appearance alone.

12. Do not fabricate NDVI, NDWI, NDBI, SAR or change values.

13. If an index or measurement is unavailable, say that it is
    unavailable rather than estimating it.

============================================================
ANSWER PRESENTATION RULES
============================================================

Return ONLY the final user-facing answer.

Use clean Markdown formatting.

Use:

## Section Heading

for major sections.

Use:

### Subheading

when a subsection is useful.

Use **bold text** for important terms, findings and values.

Use bullet points when presenting multiple observations.

Use Markdown tables when comparing numerical or categorical
information.

Keep paragraphs short and readable.

Do not put the answer inside a code block.

Do not use unnecessary emojis.

Do not repeat the user's question.

Do not add a generic introduction such as "Sure, I can help."

The answer should be concise but informative.

============================================================
OBSERVATION VS INTERPRETATION
============================================================

Prefer language such as:

"Visible evidence indicates..."

"The imagery shows..."

"The computational analysis reports..."

"This may indicate..."

"This is consistent with..."

Avoid unsupported certainty.

============================================================
CONFIDENCE
============================================================

End the answer with:

## Confidence & Uncertainty

**Confidence:** High / Medium / Low

Then provide one concise uncertainty statement.

============================================================
FINAL CONCLUSION
============================================================

Always finish with:

## Conclusion

Provide a concise final interpretation based only on available
visual, computational and geospatial evidence.
"""


# ============================================================
# TASK ROUTER
# ============================================================

def route_task(
    question: str,
    image_count: int,
    analysis_mode: str,
    modality: str,
):
    """
    Route the request to the most appropriate SatQuery AI task.
    """

    q = (question or "").lower()
    mode = (analysis_mode or "automatic").lower()
    modality = (modality or "").lower()

    # --------------------------------------------------------
    # Explicit modes
    # --------------------------------------------------------

    if mode == "change":
        return "change_analysis"

    if mode == "optical-sar":
        return "optical_sar"

    if mode == "single":

        if any(
            phrase in q
            for phrase in [
                "describe",
                "land cover",
                "land-cover",
                "scene",
                "caption",
                "what is in this image",
            ]
        ):
            return "captioning"

        return "single_vqa"

    # --------------------------------------------------------
    # Automatic routing
    # --------------------------------------------------------

    if image_count >= 2:

        if (
            "sar" in q
            or "optical" in q
            or "cross-modal" in q
            or "radar" in q
            or modality == "optical-sar"
        ):
            return "optical_sar"

        return "change_analysis"

    # --------------------------------------------------------
    # Captioning
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "describe",
            "land cover",
            "land-cover",
            "scene",
            "caption",
            "what is in this image",
        ]
    ):
        return "captioning"

    return "single_vqa"


# ============================================================
# LOCATION HELPERS
# ============================================================

def _extract_location_from_geospatial(
    geospatial: dict,
) -> dict:

    if not isinstance(geospatial, dict):
        return {
            "available": False,
            "latitude": None,
            "longitude": None,
            "location": None,
        }

    center = geospatial.get(
        "center_latitude_longitude"
    )

    if not isinstance(center, dict):
        return {
            "available": False,
            "latitude": None,
            "longitude": None,
            "location": None,
        }

    latitude = center.get("latitude")
    longitude = center.get("longitude")

    if latitude is None or longitude is None:
        return {
            "available": False,
            "latitude": None,
            "longitude": None,
            "location": None,
        }

    return {
        "available": True,
        "latitude": latitude,
        "longitude": longitude,
        "location": f"{latitude}, {longitude}",
    }


def _geospatial_context(
    geospatial: dict,
) -> dict:

    location = _extract_location_from_geospatial(
        geospatial
    )

    return {
        "available": location["available"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "location": location["location"],
        "crs": geospatial.get("crs"),
        "coordinate_system_type": geospatial.get(
            "coordinate_system_type"
        ),
        "bounds": geospatial.get("bounds"),
        "wgs84_bounds": geospatial.get(
            "wgs84_bounds"
        ),
        "resolution": geospatial.get(
            "resolution"
        ),
        "scene_area": geospatial.get(
            "scene_area"
        ),
    }


# ============================================================
# COMPACT EVIDENCE BUILDER
# ============================================================

def _compact_value(value, max_items=6):
    """
    Prevent very large arrays/dictionaries from being sent
    to the language model.
    """

    if isinstance(value, dict):

        result = {}

        for index, (key, item) in enumerate(value.items()):

            if index >= max_items:
                break

            result[key] = _compact_value(
                item,
                max_items=max_items,
            )

        return result

    if isinstance(value, list):

        return [
            _compact_value(
                item,
                max_items=max_items,
            )
            for item in value[:max_items]
        ]

    if isinstance(value, tuple):

        return [
            _compact_value(
                item,
                max_items=max_items,
            )
            for item in value[:max_items]
        ]

    if isinstance(value, (str, int, float, bool)):

        if isinstance(value, str):
            return value[:500]

        return value

    return str(value)[:500]


def _build_llm_evidence(
    computational_evidence: list[dict],
    task: str,
):
    """
    Create a compact representation of computational evidence
    specifically for the language model.
    """

    compact = []

    for item in computational_evidence:

        metadata = item.get(
            "input_metadata",
            {},
        )

        geo = item.get(
            "geospatial_context",
            {},
        )

        entry = {
            "filename": metadata.get(
                "filename"
            ),

            "dimensions": metadata.get(
                "dimensions"
            ),

            "format": metadata.get(
                "format"
            ),

            "crs": metadata.get(
                "crs"
            ),

            "latitude": metadata.get(
                "latitude"
            ),

            "longitude": metadata.get(
                "longitude"
            ),

            "location": metadata.get(
                "location"
            ),

            "geospatial_available": geo.get(
                "available",
                False,
            ),
        }

        # ----------------------------------------------------
        # Selected raster information
        # ----------------------------------------------------

        raster = item.get(
            "raster_analysis",
            {},
        )

        if isinstance(raster, dict):

            selected_raster = {}

            for key in [
                "statistics",
                "spectral_indices",
                "bands",
                "mean",
                "min",
                "max",
                "std",
                "geospatial",
                "band_availability",
                "semantic_indicators",
                "sar_analysis",
            ]:

                if key in raster:
                    selected_raster[key] = _compact_value(
                        raster[key]
                    )

            if selected_raster:
                entry["raster_analysis"] = selected_raster

        # ----------------------------------------------------
        # Encoder information
        # ----------------------------------------------------

        encoder = item.get(
            "remote_sensing_encoder",
            {},
        )

        if encoder:

            entry["remote_sensing_encoder"] = (
                _compact_value(
                    encoder,
                    max_items=4,
                )
            )

        compact.append(entry)

    # --------------------------------------------------------
    # Specialist output
    # --------------------------------------------------------

    for item in computational_evidence:

        specialist = item.get(
            "specialist_tool_output"
        )

        if specialist:

            compact.append(
                {
                    "specialist_tool_output":
                        _compact_value(
                            specialist,
                            max_items=8,
                        )
                }
            )

    return compact


# ============================================================
# MARKDOWN RESPONSE INSTRUCTIONS
# ============================================================

def _markdown_format_instruction(
    task: str,
) -> str:

    if task == "change_analysis":

        return """
============================================================
REQUIRED RESPONSE FORMAT
============================================================

## Change Overview

Provide 2-3 concise sentences describing the overall
before-versus-after situation.

## Key Changes

Use bullet points.

- **Change:** Description
- **Change:** Description
- **Change:** Description

Only include meaningful changes supported by the imagery.

## Spatial Distribution

Explain where the changes occur using relative positions
such as north, south, east, west, centre, upper portion,
lower portion, left or right side when visually justified.

## Change Interpretation

Explain what the observed changes may represent.

Clearly distinguish visible evidence from interpretation.

## Geospatial Context

If coordinates and CRS are available, provide:

| Property | Value |
|---|---|
| Latitude | ... |
| Longitude | ... |
| CRS | ... |

If unavailable, explicitly state that geospatial coordinates
are not available from the supplied metadata.

## Change Evidence

Mention computational change evidence when available.

Do not assume that every change-map pixel represents a
semantic land-cover change.

## Confidence & Uncertainty

**Confidence:** High / Medium / Low

Provide one concise uncertainty statement.

## Conclusion

Summarize the most important observed change in 1-2 sentences.
"""

    if task == "optical_sar":

        return """
============================================================
REQUIRED RESPONSE FORMAT
============================================================

## Scene Overview

Provide a concise overview of the supplied optical and SAR
imagery.

## Optical Findings

Use bullet points.

- **Vegetation:** ...
- **Water:** ...
- **Built-up / Infrastructure:** ...
- **Land Cover:** ...

Only mention categories supported by the imagery.

## SAR Findings

Explain radar-related observations.

- **Backscatter:** ...
- **Texture / Structure:** ...
- **Surface Characteristics:** ...

Do not interpret SAR grayscale as optical colour.

## Cross-Modal Insights

Explain what becomes clearer when optical and SAR evidence
are considered together.

## Spectral / Computational Evidence

If spectral indices or raster statistics are available,
present the most relevant values.

Use a table when appropriate:

| Metric | Value | Interpretation |
|---|---:|---|
| NDVI | ... | ... |
| NDWI | ... | ... |
| NDBI | ... | ... |

Do not fabricate unavailable values.

## Geospatial Context

Present available CRS, coordinates, resolution or scene area.

## Confidence & Uncertainty

**Confidence:** High / Medium / Low

Provide one concise uncertainty statement.

## Conclusion

Give a concise combined optical-SAR interpretation.
"""

    if task == "captioning":

        return """
============================================================
REQUIRED RESPONSE FORMAT
============================================================

## Scene Overview

Provide a concise description of the satellite scene.

## Key Findings

Use bullet points with bold category names.

- **Dominant Land Cover:** ...
- **Vegetation:** ...
- **Water:** ...
- **Buildings / Urban Areas:** ...
- **Roads / Infrastructure:** ...
- **Open Land / Agriculture:** ...

Only include categories that are actually supported.

## Spatial Organization

Describe how the major features are distributed spatially.

## Geospatial Context

Mention available CRS, coordinates, resolution or scene area.

Do not infer a city or country from visual appearance alone.

## Confidence & Uncertainty

**Confidence:** High / Medium / Low

Provide one concise uncertainty statement.

## Conclusion

Summarize the overall scene in 1-2 sentences.
"""

    # --------------------------------------------------------
    # Single VQA
    # --------------------------------------------------------

    return """
============================================================
REQUIRED RESPONSE FORMAT
============================================================

## Direct Answer

Answer the user's question directly in 1-3 sentences.

## Key Findings

Use concise bullet points.

- **Finding:** ...
- **Finding:** ...
- **Finding:** ...

Only include findings relevant to the question.

## Supporting Evidence

Explain the visual or computational evidence supporting
the answer.

## Spectral / Computational Insights

If NDVI, NDWI, NDBI, raster statistics or other computational
evidence is available and relevant, present it clearly.

Use a table when useful.

| Metric | Value | Interpretation |
|---|---:|---|
| NDVI | ... | ... |

Do not fabricate unavailable values.

## Geospatial Context

Mention CRS, coordinates, resolution, scene area or bounds
only when available.

## Confidence & Uncertainty

**Confidence:** High / Medium / Low

Provide one concise uncertainty statement.

## Conclusion

Give a short final interpretation.
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(
    question: str,
    task: str,
    image_count: int,
    computational_evidence: list[dict],
):
    """
    Build the final model prompt using compact computational
    evidence and professional Markdown response instructions.
    """

    # ========================================================
    # TASK-SPECIFIC INSTRUCTIONS
    # ========================================================

    if task == "change_analysis":

        task_instruction = """
Compare IMAGE 1 (BEFORE) with IMAGE 2 (AFTER).

Focus only on meaningful spatial changes.

Identify, when supported:

- new features
- disappearing features
- expansion or reduction
- vegetation change
- water change
- construction or infrastructure change
- land-use or surface change

Use computational change evidence as supporting evidence.

Do not treat every highlighted change-map pixel as a semantic
land-cover change.

Compare corresponding spatial regions.

Only make claims supported by imagery or computational evidence.
"""

    elif task == "single_vqa":

        task_instruction = """
Perform remote-sensing visual question answering.

Answer the user's question directly.

Use:

- visible image evidence
- raster statistics
- spectral indices
- geospatial metadata
- remote-sensing encoder evidence

Only use evidence that is actually available.
"""

    elif task == "captioning":

        task_instruction = """
Generate a concise but informative remote-sensing scene
description.

Cover relevant visible features such as:

- dominant land cover
- vegetation
- water
- roads
- buildings
- urban areas
- agriculture
- open land
- spatial organization

Do not infer a city or country from visual appearance alone.
"""

    elif task == "optical_sar":

        task_instruction = """
Jointly interpret optical and SAR observations.

OPTICAL:

- spectral appearance
- vegetation
- water
- land-cover patterns
- built-up areas
- roads and infrastructure

SAR:

- radar backscatter
- texture
- structural information
- surface roughness
- moisture-related evidence where justified

Never interpret SAR grayscale as normal optical colour.

Explain the complementary information provided by the two
modalities.

Do not fabricate SAR polarization or sensor information.
"""

    else:

        task_instruction = """
Analyze the supplied remote-sensing imagery using only
supported visual, computational and geospatial evidence.
"""

    # ========================================================
    # COMPACT EVIDENCE
    # ========================================================

    evidence = _build_llm_evidence(
        computational_evidence,
        task,
    )

    try:

        evidence_text = json.dumps(
            evidence,
            separators=(",", ":"),
            default=str,
        )

    except Exception:

        evidence_text = str(evidence)

    # ========================================================
    # MARKDOWN FORMAT
    # ========================================================

    format_instruction = _markdown_format_instruction(
        task
    )

    # ========================================================
    # FINAL PROMPT
    # ========================================================

    return f"""
{task_instruction}

============================================================
USER QUESTION
============================================================

{question}

============================================================
NUMBER OF IMAGES
============================================================

{image_count}

============================================================
COMPUTATIONAL EVIDENCE
============================================================

{evidence_text}

============================================================
GEOSPATIAL RULE
============================================================

Use latitude/longitude only when supplied by metadata.

Never invent coordinates, locations, CRS information or
physical measurements.

============================================================
ANSWER STYLE
============================================================

The answer must be:

- professional
- structured
- concise
- evidence-grounded
- easy to scan
- written in clean Markdown

Use **bold** for important terms and values.

Use bullet points for multiple findings.

Use tables for useful numerical comparisons.

Do not use unnecessary emojis.

Do not repeat the question.

Do not reveal reasoning.

{format_instruction}
"""


# ============================================================
# IMAGE COMPRESSION
# ============================================================

def prepare_vision_image(
    image_bytes: bytes,
    max_dimension: int = 768,
    quality: int = 72,
):
    """
    Resize and compress an image before sending it to the
    vision model.
    """

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = image.convert("RGB")

        # ----------------------------------------------------
        # Resize while preserving aspect ratio.
        # ----------------------------------------------------

        width, height = image.size

        largest_dimension = max(
            width,
            height,
        )

        if largest_dimension > max_dimension:

            scale = (
                max_dimension
                / largest_dimension
            )

            new_width = max(
                1,
                int(width * scale),
            )

            new_height = max(
                1,
                int(height * scale),
            )

            image = image.resize(
                (
                    new_width,
                    new_height,
                ),
                Image.Resampling.LANCZOS,
            )

        # ----------------------------------------------------
        # JPEG compression.
        # ----------------------------------------------------

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=True,
        )

        return output.getvalue()

    except Exception:

        return image_bytes


def image_to_data_url(
    image_bytes: bytes,
):
    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# CONFIDENCE EXTRACTION
# ============================================================

def extract_confidence(
    answer: str,
):

    if not answer:
        return "Medium"

    match = re.search(
        r"\*\*Confidence:\*\*\s*(High|Medium|Low)",
        answer,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).capitalize()

    match = re.search(
        r"Confidence:\s*(High|Medium|Low)",
        answer,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).capitalize()

    return "Medium"


# ============================================================
# ANSWER CLEANUP
# ============================================================

def clean_markdown_answer(
    answer: str,
):
    """
    Clean common formatting problems returned by the LLM while
    preserving Markdown.

    This does NOT convert Markdown to HTML because the frontend
    will render Markdown using ReactMarkdown.
    """

    if not answer:
        return (
            "## Analysis\n\n"
            "No analysis was returned by the model."
        )

    answer = answer.strip()

    # --------------------------------------------------------
    # Remove accidental fenced markdown blocks.
    # --------------------------------------------------------

    if answer.startswith("```markdown"):

        answer = re.sub(
            r"^```markdown\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        answer = re.sub(
            r"\s*```$",
            "",
            answer,
        )

    elif answer.startswith("```"):

        answer = re.sub(
            r"^```\s*",
            "",
            answer,
        )

        answer = re.sub(
            r"\s*```$",
            "",
            answer,
        )

    # --------------------------------------------------------
    # Normalize excessive blank lines.
    # --------------------------------------------------------

    answer = re.sub(
        r"\n{4,}",
        "\n\n",
        answer,
    )

    # --------------------------------------------------------
    # Convert old-style "Overview:" headings to Markdown.
    # This provides backward compatibility if the model ignores
    # the new formatting instructions.
    # --------------------------------------------------------

    heading_names = [
        "Overview",
        "Key Findings",
        "Spatial Evidence",
        "Spatial Distribution",
        "Land / Surface Interpretation",
        "Land / Surface Analysis",
        "Geospatial Context",
        "Change Evidence",
        "Change Interpretation",
        "Optical Findings",
        "SAR Findings",
        "Cross-Modal Insights",
        "Spectral / Computational Insights",
        "Supporting Evidence",
        "Direct Answer",
        "Confidence & Uncertainty",
        "Uncertainty",
        "Conclusion",
    ]

    for heading in heading_names:

        pattern = (
            r"(?m)^\s*"
            + re.escape(heading)
            + r"\s*:\s*$"
        )

        replacement = f"## {heading}"

        answer = re.sub(
            pattern,
            replacement,
            answer,
            flags=re.IGNORECASE,
        )

    # --------------------------------------------------------
    # Convert bullet characters into Markdown bullets.
    # --------------------------------------------------------

    answer = re.sub(
        r"(?m)^\s*[•●]\s*",
        "- ",
        answer,
    )

    return answer.strip()


# ============================================================
# METADATA CONTEXT
# ============================================================

def _metadata_context(
    item: dict,
):

    metadata = item.get(
        "metadata",
        {},
    )

    raster_analysis_data = item.get(
        "raster_analysis",
        {},
    )

    geospatial = raster_analysis_data.get(
        "geospatial",
        {},
    )

    location = _extract_location_from_geospatial(
        geospatial
    )

    return {
        "filename": item.get(
            "filename"
        ),

        "dimensions": [
            metadata.get("width"),
            metadata.get("height"),
        ],

        "format": metadata.get(
            "format"
        ),

        "geospatial": metadata.get(
            "geospatial",
            False,
        ),

        "crs": metadata.get(
            "crs"
        ),

        "bounds": metadata.get(
            "bounds"
        ),

        "wgs84_bounds": geospatial.get(
            "wgs84_bounds"
        ),

        "latitude": location.get(
            "latitude"
        ),

        "longitude": location.get(
            "longitude"
        ),

        "location": location.get(
            "location"
        ),

        "band_count": metadata.get(
            "count"
        ),

        "dtype": metadata.get(
            "dtype"
        ),
    }


# ============================================================
# LOCATION SUMMARY
# ============================================================

def _build_location_summary(
    computational_evidence: list[dict],
):

    locations = []

    for item in computational_evidence:

        raster_info = item.get(
            "raster_analysis",
            {},
        )

        geo = raster_info.get(
            "geospatial",
            {},
        )

        location = _extract_location_from_geospatial(
            geo
        )

        if location["available"]:

            locations.append(
                {
                    "filename": item.get(
                        "input_metadata",
                        {},
                    ).get(
                        "filename"
                    ),

                    "latitude": location[
                        "latitude"
                    ],

                    "longitude": location[
                        "longitude"
                    ],

                    "location": location[
                        "location"
                    ],

                    "crs": geo.get(
                        "crs"
                    ),

                    "wgs84_bounds": geo.get(
                        "wgs84_bounds"
                    ),
                }
            )

    if not locations:

        return {
            "available": False,
            "locations": [],
        }

    return {
        "available": True,
        "locations": locations,
    }


# ============================================================
# MAIN SATELLITE ANALYSIS PIPELINE
# ============================================================

def analyze_satellite_images(
    images,
    question: str,
    analysis_mode: str,
    modality: str,
    generated_dir: Path,
):

    # ========================================================
    # INPUT SAFETY
    # ========================================================

    if not images:
        raise ValueError(
            "At least one satellite image is required."
        )

    if not question:

        question = (
            "Describe the supplied remote-sensing imagery."
        )

    # ========================================================
    # AGENT ROUTING
    # ========================================================

    task = route_task(
        question=question,
        image_count=len(images),
        analysis_mode=analysis_mode,
        modality=modality,
    )

    model_info = MODEL_REGISTRY.get(
        task,
        MODEL_REGISTRY["single_vqa"],
    )

    model_name = model_info[
        "model"
    ]

    # ========================================================
    # AGENT STEP 1
    # COMPUTATIONAL ANALYSIS
    # ========================================================

    computational_evidence = []

    preview_bytes = []

    for image in images:

        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        preview, preview_meta = create_preview(
            image["bytes"],
            image["filename"],
            generated_dir,
        )

        preview_bytes.append(
            preview
        )

        # ----------------------------------------------------
        # Raster analysis
        # ----------------------------------------------------

        raster_info = raster_analysis(
            image["bytes"],
            image["filename"],
        )

        # ----------------------------------------------------
        # Remote-sensing encoder
        # ----------------------------------------------------

        encoder_info = adapter.extract_features(
            preview
        )

        # ----------------------------------------------------
        # Metadata context
        # ----------------------------------------------------

        image_item = {
            "filename": image.get(
                "filename"
            ),

            "metadata": image.get(
                "metadata",
                {},
            ),

            "raster_analysis": raster_info,
        }

        metadata_context = _metadata_context(
            image_item
        )

        # ----------------------------------------------------
        # Geospatial context
        # ----------------------------------------------------

        geospatial_context = _geospatial_context(
            raster_info.get(
                "geospatial",
                {},
            )
        )

        # ----------------------------------------------------
        # Store evidence
        # ----------------------------------------------------

        computational_evidence.append(
            {
                "input_metadata": metadata_context,

                "preview": preview_meta,

                "raster_analysis": raster_info,

                "geospatial_context": geospatial_context,

                "remote_sensing_encoder": encoder_info,
            }
        )

    # ========================================================
    # AGENT STEP 2
    # SPECIALIST TOOL EXECUTION
    # ========================================================

    evidence_image = None

    specialist_result = {}

    # --------------------------------------------------------
    # CHANGE DETECTION
    # --------------------------------------------------------

    if (
        task == "change_analysis"
        and len(images) == 2
    ):

        evidence_image, specialist_result = (
            create_change_map(
                images[0]["bytes"],
                images[0]["filename"],
                images[1]["bytes"],
                images[1]["filename"],
                generated_dir,
            )
        )

    # --------------------------------------------------------
    # OPTICAL-SAR FUSION
    # --------------------------------------------------------

    elif (
        task == "optical_sar"
        and len(images) == 2
    ):

        evidence_image, specialist_result = (
            create_optical_sar_evidence(
                images[0]["bytes"],
                images[0]["filename"],
                images[1]["bytes"],
                images[1]["filename"],
                generated_dir,
            )
        )

    # --------------------------------------------------------
    # Store specialist evidence
    # --------------------------------------------------------

    if specialist_result:

        computational_evidence.append(
            {
                "specialist_tool_output":
                    specialist_result
            }
        )

    # ========================================================
    # AGENT STEP 3
    # DOMAIN ADAPTATION
    # ========================================================

    domain_context = adapter.domain_prompt(
        task=task,
        modality=modality,
    )

    # ========================================================
    # BI-TEMPORAL DOMAIN CONTEXT
    # ========================================================

    if task == "change_analysis":

        domain_context = (
            "Remote-sensing change detection domain context. "
            "Compare spatially corresponding regions and use "
            "computational change evidence as supporting evidence."
        )

    # ========================================================
    # BUILD FINAL PROMPT
    # ========================================================

    prompt = (
        domain_context
        + "\n\n"
        + build_prompt(
            question=question,
            task=task,
            image_count=len(images),
            computational_evidence=computational_evidence,
        )
    )

    # ========================================================
    # VISION CONTENT
    # ========================================================

    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    labels = []

    for index, preview in enumerate(
        preview_bytes
    ):

        labels.append(
            f"IMAGE {index + 1}"
        )

        content.append(
            {
                "type": "text",
                "text": f"IMAGE {index + 1}:",
            }
        )

        # ----------------------------------------------------
        # Compress preview before sending to Groq.
        # ----------------------------------------------------

        if task == "change_analysis":

            vision_bytes = prepare_vision_image(
                preview,
                max_dimension=640,
                quality=65,
            )

        else:

            vision_bytes = prepare_vision_image(
                preview,
                max_dimension=768,
                quality=72,
            )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        vision_bytes
                    ),
                },
            }
        )

    # ========================================================
    # GROQ COMPLETION SIZE
    # ========================================================

    if task == "change_analysis":

        completion_tokens = 650

    elif task == "optical_sar":

        completion_tokens = 850

    else:

        completion_tokens = 1000

    # ========================================================
    # GROQ VISION CALL
    # ========================================================

    try:

        response = client.chat.completions.create(
            model=model_name,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },

                {
                    "role": "user",
                    "content": content,
                },
            ],

            temperature=0.0,

            max_completion_tokens=completion_tokens,
        )

    except Exception as exc:

        # ----------------------------------------------------
        # Make backend error understandable.
        # ----------------------------------------------------

        error_text = str(exc)

        if (
            "413" in error_text
            or "tokens per minute" in error_text.lower()
            or "rate_limit_exceeded" in error_text.lower()
        ):

            raise RuntimeError(
                "The SatQuery AI request was too large for "
                "the current Groq model limit. The imagery "
                "and computational evidence were compressed. "
                "Please use smaller images or try again."
            ) from exc

        raise

    # ========================================================
    # ANSWER
    # ========================================================

    raw_answer = (
        response
        .choices[0]
        .message
        .content
    )

    answer = clean_markdown_answer(
        raw_answer
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = extract_confidence(
        answer
    )

    # ========================================================
    # EXECUTION TRACE
    # ========================================================

    executed_tools = list(
        model_info[
            "tools"
        ]
    )

    if task == "change_analysis":

        executed_tools.append(
            "computational_change_evidence"
        )

    if task == "optical_sar":

        executed_tools.append(
            "computational_cross_modal_evidence"
        )

    # ========================================================
    # ADAPTER STATUS
    # ========================================================

    adaptation_status = adapter.status()

    # ========================================================
    # LOCATION SUMMARY
    # ========================================================

    location_summary = (
        _build_location_summary(
            computational_evidence
        )
    )

    # ========================================================
    # GEOSPATIAL AVAILABILITY
    # ========================================================

    geospatial_available = (
        location_summary[
            "available"
        ]
    )

    # ========================================================
    # EXECUTION SUMMARY
    # ========================================================

    execution_summary = {

        "selected_task": task,

        "image_count": len(
            images
        ),

        "requested_modality": modality,

        "query_routing": (
            "automatic agentic routing"
        ),

        "input_validation": "passed",

        "remote_sensing_preprocessing": (
            "executed"
        ),

        "geospatial_analysis": (
            "available"
            if geospatial_available
            else "not available"
        ),

        "specialist_tool_execution": (
            "executed"
            if specialist_result
            else "not required"
        ),

        "model_selected": model_name,

        "visual_evidence_generated": bool(
            evidence_image
        ),

        "structured_markdown_output": True,

        "report_generation_available": True,
    }

    # ========================================================
    # FINAL API RESPONSE
    # ========================================================

    return {

        "task": task,

        "models": [
            {
                "name": model_name,
                "role": model_info[
                    "description"
                ],
            }
        ],

        "tools": executed_tools,

        # ----------------------------------------------------
        # Professionally formatted Markdown answer
        # ----------------------------------------------------

        "answer": answer,

        "confidence": confidence,

        # ----------------------------------------------------
        # Geospatial result
        # ----------------------------------------------------

        "location": location_summary,

        "geospatial": {

            "available": geospatial_available,

            "locations": location_summary[
                "locations"
            ],

            "description": (
                "Latitude and longitude are derived "
                "from GeoTIFF geospatial metadata and "
                "converted to WGS84 when CRS information "
                "is available."
            ),
        },

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        "evidence": {

            "type": (
                "visual_and_computational"
            ),

            "description": (
                "Evidence combines the supplied imagery "
                "with remote-sensing raster statistics, "
                "spectral indices, geospatial metadata, "
                "optional domain-adapted visual features, "
                "and specialist change or optical-SAR evidence."
            ),

            "computational": (
                computational_evidence
            ),

            "specialist": (
                specialist_result
            ),
        },

        # ----------------------------------------------------
        # Execution trace
        # ----------------------------------------------------

        "execution_summary": (
            execution_summary
        ),

        # ----------------------------------------------------
        # Domain adaptation
        # ----------------------------------------------------

        "adaptation": {

            "status": adaptation_status.get(
                "status"
            ),

            "label_source": (
                adaptation_status.get(
                    "label_source"
                )
            ),

            "label_count": (
                adaptation_status.get(
                    "label_count"
                )
            ),

            "encoder_active": (
                adaptation_status.get(
                    "encoder_active"
                )
            ),

            "method": (
                "Remote-sensing domain adaptation "
                "through an open-source remote-sensing "
                "vocabulary layer and optional Sentinel-2 "
                "pretrained TorchGeo visual encoder."
            ),
        },

        # ----------------------------------------------------
        # Generated evidence image
        # ----------------------------------------------------

        "evidence_image": evidence_image,
    }