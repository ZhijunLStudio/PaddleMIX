import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._clip_filter import CLIPFilterConfig

def main():
    # Load dataset
    dataset = MMDataset.from_json('./my.json')
    print(f"Original dataset size: {len(dataset)}")

    # Configure CLIP filter parameters
    config = CLIPFilterConfig(
        model_name="paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K",
        threshold=0.15  # Set similarity threshold
    )

    # Process dataset with chained operations
    processed_dataset = (
        dataset
        .filter_by_clip(
            text_prompt="person",  # Replace with your desired prompt
            config=config
        )
        .nonempty()  # Remove empty items
    )

    print(f"Processed dataset size: {len(processed_dataset)}")
    
    # Export processed dataset
    processed_dataset.export_json('filtered_output_clip.json')

if __name__ == "__main__":
    main()
