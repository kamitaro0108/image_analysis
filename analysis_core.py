from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import exposure, feature, filters, measure, morphology, segmentation, util


@dataclass
class DetectionResult:
    selected: np.ndarray
    normalized: np.ndarray
    corrected: np.ndarray
    smoothed: np.ndarray
    mask: np.ndarray
    cleaned: np.ndarray
    labels: np.ndarray
    measurements: pd.DataFrame
    threshold_value: float | None


def read_tiff(source) -> np.ndarray:
    if isinstance(source, (bytes, bytearray)):
        return np.asarray(tifffile.imread(BytesIO(source)))
    return np.asarray(tifffile.imread(source))


def select_channel(image: np.ndarray, channel: str = "green", frame_index: int = 0) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[-1] in (3, 4):
        channels = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
        if channel == "max":
            return image[..., :3].max(axis=-1)
        if channel == "mean":
            return image[..., :3].mean(axis=-1)
        return image[..., channels.get(channel, 1)]

    if image.ndim == 3:
        frame_index = int(np.clip(frame_index, 0, image.shape[0] - 1))
        if channel == "max":
            return image.max(axis=0)
        if channel == "mean":
            return image.mean(axis=0)
        return image[frame_index]

    if image.ndim == 4 and image.shape[-1] in (3, 4):
        frame_index = int(np.clip(frame_index, 0, image.shape[0] - 1))
        return select_channel(image[frame_index], channel=channel)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize_float(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    image = util.img_as_float(image)
    low, high = np.percentile(image, [p_low, p_high])
    if high <= low:
        return np.zeros_like(image, dtype=float)
    return exposure.rescale_intensity(image, in_range=(low, high), out_range=(0.0, 1.0))


def preprocess_image(
    image: np.ndarray,
    channel: str = "green",
    frame_index: int = 0,
    p_low: float = 1.0,
    p_high: float = 99.8,
    background_sigma: float = 20.0,
    smooth_sigma: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = select_channel(image, channel=channel, frame_index=frame_index)
    normalized = normalize_float(selected, p_low=p_low, p_high=p_high)

    if background_sigma and background_sigma > 0:
        background = filters.gaussian(normalized, sigma=background_sigma, preserve_range=True)
        corrected = normalized - background
        corrected = exposure.rescale_intensity(corrected, out_range=(0.0, 1.0))
    else:
        corrected = normalized

    if smooth_sigma and smooth_sigma > 0:
        smoothed = filters.gaussian(corrected, sigma=smooth_sigma, preserve_range=True)
    else:
        smoothed = corrected

    return selected, normalized, corrected, smoothed


def threshold_image(
    image: np.ndarray,
    method: str = "otsu",
    manual_threshold: float = 0.25,
    local_block_size: int = 51,
    local_offset: float = 0.0,
) -> tuple[np.ndarray, float | None]:
    if method == "manual":
        threshold = float(manual_threshold)
        return image > threshold, threshold

    if method == "local":
        block_size = int(local_block_size)
        if block_size % 2 == 0:
            block_size += 1
        threshold_map = filters.threshold_local(image, block_size=block_size, offset=local_offset)
        return image > threshold_map, None

    threshold = float(filters.threshold_otsu(image))
    return image > threshold, threshold


def segment_particles(
    mask: np.ndarray,
    min_area: int = 20,
    max_area: int = 2000,
    clear_border: bool = False,
    split_touching: bool = True,
    min_distance: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=int(min_area))
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=int(min_area))
    if clear_border:
        cleaned = segmentation.clear_border(cleaned)

    if split_touching:
        distance = ndi.distance_transform_edt(cleaned)
        coords = feature.peak_local_max(distance, min_distance=int(min_distance), labels=cleaned)
        markers = np.zeros(distance.shape, dtype=int)
        if len(coords) > 0:
            markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
            markers = measure.label(markers > 0)
            labels = segmentation.watershed(-distance, markers, mask=cleaned)
        else:
            labels = measure.label(cleaned)
    else:
        labels = measure.label(cleaned)

    if max_area and max_area > 0 and labels.max() > 0:
        keep = np.zeros(labels.max() + 1, dtype=bool)
        for prop in measure.regionprops(labels):
            keep[prop.label] = int(min_area) <= prop.area <= int(max_area)
        labels = keep[labels] * labels
        labels = measure.label(labels > 0)

    return cleaned, labels


def measure_particles(labels: np.ndarray, intensity_image: np.ndarray) -> pd.DataFrame:
    columns = [
        "label",
        "area",
        "centroid_y",
        "centroid_x",
        "mean_intensity",
        "max_intensity",
        "equivalent_diameter",
    ]
    if labels.max() == 0:
        return pd.DataFrame(columns=columns)

    table = measure.regionprops_table(
        labels,
        intensity_image=intensity_image,
        properties=("label", "area", "centroid", "mean_intensity", "max_intensity", "equivalent_diameter_area"),
    )
    df = pd.DataFrame(table)
    return df.rename(
        columns={
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
            "equivalent_diameter_area": "equivalent_diameter",
        }
    )


def detect_particles(image: np.ndarray, **params) -> DetectionResult:
    selected, normalized, corrected, smoothed = preprocess_image(
        image,
        channel=params.get("channel", "green"),
        frame_index=params.get("frame_index", 0),
        p_low=params.get("p_low", 1.0),
        p_high=params.get("p_high", 99.8),
        background_sigma=params.get("background_sigma", 20.0),
        smooth_sigma=params.get("smooth_sigma", 1.2),
    )
    mask, threshold_value = threshold_image(
        smoothed,
        method=params.get("threshold_method", "otsu"),
        manual_threshold=params.get("manual_threshold", 0.25),
        local_block_size=params.get("local_block_size", 51),
        local_offset=params.get("local_offset", 0.0),
    )
    cleaned, labels = segment_particles(
        mask,
        min_area=params.get("min_area", 20),
        max_area=params.get("max_area", 2000),
        clear_border=params.get("clear_border", False),
        split_touching=params.get("split_touching", True),
        min_distance=params.get("min_distance", 5),
    )
    measurements = measure_particles(labels, corrected)
    return DetectionResult(
        selected=selected,
        normalized=normalized,
        corrected=corrected,
        smoothed=smoothed,
        mask=mask,
        cleaned=cleaned,
        labels=labels,
        measurements=measurements,
        threshold_value=threshold_value,
    )


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8 and image.ndim == 3:
        return image
    image = util.img_as_float(image)
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def make_overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    base = image_to_uint8(exposure.rescale_intensity(image, out_range=(0.0, 1.0)))
    rgb = np.repeat(base[..., None], 3, axis=-1)
    boundaries = segmentation.find_boundaries(labels, mode="outer")
    rgb[boundaries] = np.array([255, 230, 0], dtype=np.uint8)
    return rgb


def labels_to_tiff_bytes(labels: np.ndarray) -> bytes:
    buffer = BytesIO()
    tifffile.imwrite(buffer, labels.astype(np.uint16))
    return buffer.getvalue()
