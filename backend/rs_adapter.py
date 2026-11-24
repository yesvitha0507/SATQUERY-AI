from __future__ import annotations

from pathlib import Path
import os
import re
import io

import numpy as np
from PIL import Image


# ============================================================
# REMOTE SENSING VOCABULARY
# ============================================================

DEFAULT_RS_CLASSES = [
    "continuous urban fabric",
    "discontinuous urban fabric",
    "industrial or commercial units",
    "road and rail networks",
    "port areas",
    "airports",
    "construction sites",
    "green urban areas",
    "arable land",
    "permanently irrigated land",
    "rice fields",
    "vineyards",
    "fruit trees and berry plantations",
    "pastures",
    "complex cultivation patterns",
    "agro-forestry areas",
    "forests",
    "natural grassland",
    "transitional woodland shrub",
    "water bodies",
    "coastal waters",
    "beaches and dunes",
    "bare rocks",
    "sparsely vegetated areas",
    "glaciers and perpetual snow",
]


# ============================================================
# LOAD BIGEARTHNET LABELS
# ============================================================

def _load_open_source_labels() -> tuple[list[str], str]:

    candidates = []

    env_path = os.getenv(
        "BIGEARTHNET_LABEL_FILE"
    )

    if env_path:
        candidates.append(
            Path(env_path)
        )

    candidates.extend(
        [
            Path(__file__).resolve().parent
            / "data"
            / "BigEarthNet.txt",

            Path(__file__).resolve().parent
            / "BigEarthNet.txt",
        ]
    )

    for path in candidates:

        try:

            if not path.exists():
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            labels = []

            for line in text.splitlines():

                line = line.strip()

                if not line:
                    continue

                # Remove numbering.
                line = re.sub(
                    r"^\d+\s*[\.\):-]\s*",
                    "",
                    line,
                )

                # Remove bullets.
                line = re.sub(
                    r"^[\-\*\u2022]\s*",
                    "",
                    line,
                )

                if 2 <= len(line) <= 120:
                    labels.append(line)

            labels = list(
                dict.fromkeys(labels)
            )

            if labels:

                return (
                    labels[:200],
                    str(path),
                )

        except Exception:
            continue

    return (
        DEFAULT_RS_CLASSES,
        "built-in remote-sensing vocabulary",
    )


# ============================================================
# REMOTE SENSING ADAPTER
# ============================================================

class RemoteSensingAdapter:
    """
    Optional remote-sensing domain adapter.

    Features:
    - BigEarthNet-compatible vocabulary
    - Optional TorchGeo Sentinel-2 encoder
    - Lightweight feature extraction
    - Domain-specific prompt context
    """

    def __init__(self):

        self.labels, self.label_source = (
            _load_open_source_labels()
        )

        self.encoder = None
        self.transform = None

        self.device = "cpu"

        self.encoder_status = (
            "Remote-sensing vocabulary adapter active"
        )

        self._try_load_torchgeo_encoder()

    # ========================================================
    # TORCHGEO
    # ========================================================

    def _try_load_torchgeo_encoder(self):

        try:

            import torch
            from torchgeo.models import (
                ResNet50_Weights,
                resnet50,
            )

            weights = (
                ResNet50_Weights.SENTINEL2_RGB_MOCO
            )

            self.encoder = resnet50(
                weights=weights
            )

            self.encoder.eval()

            self.transform = (
                weights.transforms()
            )

            self.device = "cpu"

            self.encoder_status = (
                "Sentinel-2 remote-sensing "
                "pretrained visual encoder active"
            )

        except Exception as error:

            self.encoder = None
            self.transform = None

            self.encoder_status = (
                "Remote-sensing vocabulary adapter active; "
                "optional TorchGeo encoder unavailable: "
                f"{type(error).__name__}"
            )

    # ========================================================
    # DOMAIN PROMPT
    # ========================================================

    def domain_prompt(
        self,
        task: str,
        modality: str,
    ) -> str:

        selected = self.labels[:60]

        ontology = ", ".join(
            selected
        )

        return f"""
REMOTE-SENSING DOMAIN ADAPTATION CONTEXT

Task:
{task}

Requested modality:
{modality}

Remote-sensing scene ontology:
{ontology}

Interpretation rules:

1. Optical imagery
   - Use visible colour, spectral response,
     vegetation patterns, water response and
     spatial context.

2. Multispectral imagery
   - Consider spectral bands and derived indices
     such as NDVI, NDWI and NDBI when available.

3. SAR imagery
   - Interpret radar backscatter, texture,
     polarization and structural response.
   - Do not interpret SAR intensity as ordinary
     optical colour.

4. Multitemporal imagery
   - Treat before/after images as observations
     acquired at different times.
   - Look for spatially consistent changes.

5. Optical + SAR
   - Treat the sensors as complementary.
   - Optical contributes spectral/contextual evidence.
   - SAR contributes structural/backscatter evidence.

6. Geospatial reasoning
   - Use CRS, bounds, resolution and projected
     coordinates when available.

7. Reliability
   - Do not invent coordinates.
   - Do not claim exact geographic locations unless
     supported by metadata.
   - Do not treat visual differences automatically
     as semantic land-cover changes.

The system includes:
- Remote-sensing domain vocabulary
- Optional BigEarthNet-compatible labels
- Optional Sentinel-2 pretrained TorchGeo encoder
"""

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def extract_features(
        self,
        image_bytes: bytes,
    ) -> dict:

        if (
            self.encoder is None
            or self.transform is None
        ):

            return {
                "active": False,
                "status": self.encoder_status,
                "dimension": None,
                "feature_norm": None,
            }

        try:

            import torch

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            transformed = (
                self.transform(image)
                .unsqueeze(0)
            )

            with torch.no_grad():

                features = self.encoder(
                    transformed
                )

            vector = (
                features
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            norm = (
                np.linalg.norm(vector)
                + 1e-8
            )

            normalized = (
                vector / norm
            )

            return {
                "active": True,
                "status": self.encoder_status,
                "dimension": int(
                    normalized.shape[0]
                ),
                "feature_norm": round(
                    float(
                        np.linalg.norm(
                            normalized
                        )
                    ),
                    4,
                ),
            }

        except Exception as error:

            return {
                "active": False,
                "status": (
                    "Encoder initialized but "
                    "feature extraction failed: "
                    f"{type(error).__name__}"
                ),
                "dimension": None,
                "feature_norm": None,
            }

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict:

        return {
            "status": self.encoder_status,
            "label_source": self.label_source,
            "label_count": len(
                self.labels
            ),
            "encoder_active": (
                self.encoder is not None
            ),
            "device": self.device,
        }


adapter = RemoteSensingAdapter()