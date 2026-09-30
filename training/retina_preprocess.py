import io

import numpy as np

from PIL import Image


ALLOWED_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP"
}


MIN_WIDTH = 224
MIN_HEIGHT = 224


def load_image(image_bytes: bytes) -> Image.Image:

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        original_format = image.format

        if original_format not in ALLOWED_FORMATS:

            raise ValueError(
                f"Unsupported image format: {original_format}"
            )

        image.verify()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = image.convert("RGB")

        if (
            image.width < MIN_WIDTH
            or image.height < MIN_HEIGHT
        ):

            raise ValueError(
                "Image resolution is too low"
            )

        return image

    except Exception as e:

        raise ValueError(
            f"Invalid retinal image: {e}"
        )


def pil_to_numpy(image: Image.Image):

    return np.array(image)