import os
from PIL import Image
from typing import Optional, List
from ...core import MMDataset, register
from functools import partial


# Define the resolution filter function
def is_valid_image_resolution(
    item,
    min_width: float = 112,
    min_height: float = 112,
    max_width: Optional[float] = None,
    max_height: Optional[float] = None
) -> bool:
    """
    Checks whether the image resolution is within the specified minimum and maximum width/height range.

    Args:
        item (dict): A dictionary containing the image path and related information.
        min_width (float): Minimum width, default is 112.
        min_height (float): Minimum height, default is 112.
        max_width (Optional[float]): Maximum width, default is None (no limit).
        max_height (Optional[float]): Maximum height, default is None (no limit).

    Returns:
        bool: True if the image resolution meets the criteria; otherwise, False.
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return False

    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # Check minimum width and height
            if width < min_width or height < min_height:
                return False

            # Check maximum width and height
            if max_width is not None and width > max_width:
                return False
            if max_height is not None and height > max_height:
                return False

            return True
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False


@register()
def image_resolution_filter(
    dataset,
    min_width: Optional[float] = 112,
    min_height: Optional[float] = 112,
    max_width: Optional[float] = None,
    max_height: Optional[float] = None
) -> MMDataset:
    print("Filtering images with invalid resolutions...")
    # Create the filter function
    filter_func = partial(
        is_valid_image_resolution,
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height
    )
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func,
        max_workers=8,
        progress=True
    )
    
    return filtered_dataset