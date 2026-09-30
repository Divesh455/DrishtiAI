import numpy as np

from PIL import Image

from backend.models.quality_checker import (
    ImageQualityChecker
)


class QualityService:

    def __init__(self):

        self.checker = ImageQualityChecker()

    def check_pil_image(
        self,
        image: Image.Image
    ):

        image = image.convert(
            "RGB"
        )

        image_array = np.array(
            image
        )

        result = self.checker.analyze(
            image_array
        )

        result["recommendation"] = (
            self._generate_recommendation(
                result
            )
        )

        return result

    def _generate_recommendation(
        self,
        result
    ):

        if result["status"] == "good":

            return (
                "Image quality is sufficient "
                "for screening."
            )

        checks = result["checks"]

        if not checks["resolution"]["passed"]:

            return (
                "Image resolution is too low. "
                "Please capture a higher-resolution "
                "retinal image."
            )

        if not checks["blur"]["passed"]:

            return (
                "Image is too blurry. "
                "Hold the camera steady and "
                "capture another retinal image."
            )

        if not checks["brightness"]["passed"]:

            brightness = checks[
                "brightness"
            ]["brightness"]

            if brightness < 35:

                return (
                    "Image is too dark. "
                    "Improve illumination and "
                    "capture another image."
                )

            return (
                "Image is too bright. "
                "Reduce excessive illumination "
                "and capture another image."
            )

        if not checks["contrast"]["passed"]:

            return (
                "Image contrast is too low. "
                "Please capture a clearer "
                "retinal image."
            )

        if not checks[
            "retina_visibility"
        ]["passed"]:

            return (
                "Retinal region is not clearly "
                "visible. Please reposition "
                "the camera and capture the "
                "retina again."
            )

        return (
            "Image quality is insufficient. "
            "Please capture another retinal image."
        )