from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


# ============================================================
# U-NET BUILDING BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_classes=5
    ):

        super().__init__()

        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        self.pool = nn.MaxPool2d(
            2,
            2
        )

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            2,
            2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        self.output = nn.Conv2d(
            64,
            num_classes,
            1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# LESION DETECTOR
# ============================================================

class LesionDetector:

    CLASS_NAMES = {
        1: "microaneurysm",
        2: "hemorrhage",
        3: "hard_exudate",
        4: "soft_exudate"
    }

    def __init__(
        self,
        model_path="weights/lesion_model.pth",
        image_size=512,
        device=None
    ):

        self.model_path = Path(model_path)

        self.image_size = image_size

        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None

        self._load_model()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Lesion model not found: "
                f"{self.model_path}"
            )

        self.model = UNet(
            in_channels=3,
            num_classes=5
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    # ========================================================
    # PREPROCESS IMAGE
    # ========================================================

    def _preprocess(self, image):

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        resized = cv2.resize(
            image_rgb,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_AREA
        )

        resized = (
            resized.astype(np.float32)
            / 255.0
        )

        resized = np.transpose(
            resized,
            (2, 0, 1)
        )

        tensor = torch.tensor(
            resized,
            dtype=torch.float32
        )

        tensor = tensor.unsqueeze(0)

        return tensor.to(
            self.device
        )

    # ========================================================
    # GET CONNECTED REGIONS
    # ========================================================

    def _get_regions(
        self,
        mask,
        class_id,
        min_area=5
    ):

        binary = (
            mask == class_id
        ).astype(np.uint8)

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        regions = []

        for region_id in range(
            1,
            num_labels
        ):

            x = int(
                stats[
                    region_id,
                    cv2.CC_STAT_LEFT
                ]
            )

            y = int(
                stats[
                    region_id,
                    cv2.CC_STAT_TOP
                ]
            )

            width = int(
                stats[
                    region_id,
                    cv2.CC_STAT_WIDTH
                ]
            )

            height = int(
                stats[
                    region_id,
                    cv2.CC_STAT_HEIGHT
                ]
            )

            area = int(
                stats[
                    region_id,
                    cv2.CC_STAT_AREA
                ]
            )

            if area < min_area:
                continue

            center_x = float(
                centroids[
                    region_id,
                    0
                ]
            )

            center_y = float(
                centroids[
                    region_id,
                    1
                ]
            )

            regions.append(
                {
                    "bbox": [
                        x,
                        y,
                        x + width,
                        y + height
                    ],
                    "area_pixels": area,
                    "center": [
                        round(center_x, 2),
                        round(center_y, 2)
                    ]
                }
            )

        return regions

    # ========================================================
    # DETECT
    # ========================================================

    def detect(self, image):

        if image is None:

            raise ValueError(
                "Input image is None."
            )

        original_height, original_width = (
            image.shape[:2]
        )

        input_tensor = self._preprocess(
            image
        )

        with torch.no_grad():

            output = self.model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            )[0]

            # Maximum class confidence for each pixel
            pixel_confidence = torch.max(
                probabilities,
                dim=1
            )[0][0]

        prediction = (
            prediction
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        pixel_confidence = (
            pixel_confidence
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Convert coordinates back to original image size
        # ----------------------------------------------------

        scale_x = (
            original_width
            / self.image_size
        )

        scale_y = (
            original_height
            / self.image_size
        )

        lesions = []

        for class_id, class_name in (
            self.CLASS_NAMES.items()
        ):

            regions = self._get_regions(
                prediction,
                class_id
            )

            for region in regions:

                x1, y1, x2, y2 = (
                    region["bbox"]
                )

                # --------------------------------------------------------
                # Confidence for this lesion region
                # --------------------------------------------------------

                region_mask = (
                    prediction[
                        y1:y2,
                        x1:x2
                    ] == class_id
                )

                region_confidence_map = (
                    pixel_confidence[
                        y1:y2,
                        x1:x2
                    ]
                )

                if np.any(region_mask):

                    lesion_confidence = float(
                        np.mean(
                            region_confidence_map[
                                region_mask
                            ]
                        )
                    )

                else:

                    lesion_confidence = 0.0

                # --------------------------------------------------------
                # Convert bounding box back to original image
                # coordinates
                # --------------------------------------------------------

                original_x1 = int(
                    x1 * scale_x
                )

                original_y1 = int(
                    y1 * scale_y
                )

                original_x2 = int(
                    x2 * scale_x
                )

                original_y2 = int(
                    y2 * scale_y
                )

                lesions.append(
                    {
                        "type": class_name,

                        "class_id": class_id,

                        "confidence": round(
                            lesion_confidence,
                            4
                        ),

                        "bbox": [
                            original_x1,
                            original_y1,
                            original_x2,
                            original_y2
                        ],

                        "area_pixels_512": (
                            region[
                                "area_pixels"
                            ]
                        )
                    }
                )

        return {
            "status": "success",
            "image_size": {
                "width": original_width,
                "height": original_height
            },
            "lesion_count": len(lesions),
            "lesions": lesions
        }


# ============================================================
# CREATE MODEL
# ============================================================

def create_lesion_model(
    num_classes=5,
    device=None
):

    model = UNet(
        in_channels=3,
        num_classes=num_classes
    )

    if device is not None:

        model = model.to(
            device
        )

    return model