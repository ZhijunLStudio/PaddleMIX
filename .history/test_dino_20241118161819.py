import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._grounding_dino_filter import GroundingDinoConfig

def main():
    # Load dataset
    dataset = MMDataset.from_json('./my.json')
    print(f"Original dataset size: {len(dataset)}")

    # Configure Grounding DINO parameters
    config = GroundingDinoConfig(
        box_threshold: float = 0.3
        min_objects: int = 1
        max_objects: int = 4
        min_aspect_ratio: float = 0.05  # 5%
        max_aspect_ratio: float = 0.95  # 95%

    )

    # Process dataset with chained operations
    processed_dataset = (
        dataset
        .filter_by_dino(
            prompt="", 
            # config=config
        )
        .nonempty()  # 移除空值
    )

    print(f"Processed dataset size: {len(processed_dataset)}")
    
    # Export processed dataset
    processed_dataset.export_json('filtered_output.json')

if __name__ == "__main__":
    main()