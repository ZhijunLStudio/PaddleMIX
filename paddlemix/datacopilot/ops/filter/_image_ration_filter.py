import os
from PIL import Image
from typing import Optional
from ...core import MMDataset, register
from functools import partial


# Define the aspect ratio filter function
def is_valid_image_aspect_ratio(
    item,
    min_ratio: float = 0.333,
    max_ratio: float = 3.0
) -> bool:
    """
    Checks whether the aspect ratio of the image is within the given range.

    Args:
        item (dict): A dictionary containing the image path and related information.
        min_ratio (float): Minimum aspect ratio, default is 0.333.
        max_ratio (float): Maximum aspect ratio, default is 3.0.

    Returns:
        bool: True if the aspect ratio is within the range; otherwise, False.
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return False

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            ratio = width / height
            return min_ratio <= ratio <= max_ratio
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False


@register()
def image_ration_filter(
    dataset,
    min_ratio: Optional[float] = 0.333,
    max_ratio: Optional[float] = 3.0
) -> MMDataset:
    print("Filtering images with invalid aspect ratios...")
    # Create the filter function
    filter_func = partial(is_valid_image_aspect_ratio, min_ratio=min_ratio, max_ratio=max_ratio)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func,
        max_workers=8,
        progress=True
    )
    
    return filtered_dataset