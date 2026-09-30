from pathlib import Path
import sys

import cv2
import numpy as np

from fastapi import APIRouter, File, UploadFile, HTTPException


# =============================================================
# PROJECT ROOT
# =============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


# =============================================================
# PROJECT IMPORTS
# =============================================================

from backend.services.screening_service import ScreeningService
from backend.models.quality_checker import ImageQualityChecker


# =============================================================
# ROUTER
# =============================================================

router = APIRouter(
    prefix="/api/v1/screening",
    tags=["Screening"]
)


# =============================================================
# ALLOWED IMAGE FORMATS
# =============================================================

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


# =============================================================
# SCREENING SERVICE
# =============================================================

screening_service = ScreeningService()


# =============================================================
# IMAGE QUALITY CHECKER
# =============================================================

quality_checker = ImageQualityChecker()


# =============================================================
# HEALTH / STATUS
# =============================================================

@router.get("/status")
def screening_status():

    return {
        "service": "DrishtiAI Screening",
        "status": "ready",
        "components": {
            "image_quality": "ready",
            "dr_classifier": "ready",
            "gradcam": "ready",
            "lesion_detector": "ready",
            "evidence_service": "ready"
        }
    }


# =============================================================
# QUALITY CHECK
# =============================================================

@router.post("/quality")
async def check_image_quality(
    file: UploadFile = File(...)
):

    # ---------------------------------------------------------
    # Validate content type
    # ---------------------------------------------------------

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPEG, PNG or WEBP."
            )
        )

    # ---------------------------------------------------------
    # Read image bytes
    # ---------------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    # ---------------------------------------------------------
    # Decode image
    # ---------------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Could not decode uploaded image."
        )

    # ---------------------------------------------------------
    # Convert BGR -> RGB
    # ---------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ---------------------------------------------------------
    # Run quality checker
    # ---------------------------------------------------------

    try:

        result = quality_checker.analyze(
            image_rgb
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Image quality analysis failed: "
                f"{str(error)}"
            )
        )

    # ---------------------------------------------------------
    # Response
    # ---------------------------------------------------------

    return {
        "filename": file.filename,
        "result": result
    }


# =============================================================
# COMPLETE AI SCREENING
# =============================================================

@router.post("/screen")
async def screen_retinal_image(
    file: UploadFile = File(...)
):

    # ---------------------------------------------------------
    # Validate content type
    # ---------------------------------------------------------

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPEG, PNG or WEBP."
            )
        )

    # ---------------------------------------------------------
    # Read image
    # ---------------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    # ---------------------------------------------------------
    # Decode image
    # ---------------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Could not decode uploaded image."
        )

    # ---------------------------------------------------------
    # Run complete pipeline
    # ---------------------------------------------------------

    try:

        result = screening_service.screen(
            image
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"AI screening failed: "
                f"{str(error)}"
            )
        )

    # ---------------------------------------------------------
    # Response
    # ---------------------------------------------------------

    return {
        "filename": file.filename,
        "result": result
    }