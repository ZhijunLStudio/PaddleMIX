import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._grounding_dino_filter import GroundingDinoConfig

def main():
    # Load dataset
    dataset = MMDataset.from_json('./my.json')
    print(f"Original dataset size: {len(dataset)}")

    # Configure Grounding DINO parameters
    config = GroundingDinoConfig(
        box_threshold=0.1,  # 调整对象检测的置信度阈值
        text_threshold=0.1, # 文本匹配阈值
        min_objects=1,       # 最少检测对象数
        max_objects=5,       # 最多检测对象数
        min_aspect_ratio=0.1,# 最小宽高比
        max_aspect_ratio=2.0 # 最大宽高比
    )

    # Process dataset with chained operations
    processed_dataset = (
        dataset
        .filter_by_dino(
            prompt="", 
            config=config
        )
        .nonempty()  # 移除空值
    )

    print(f"Processed dataset size: {len(processed_dataset)}")
    
    # Export processed dataset
    processed_dataset.export_json('filtered_output.json')

if __name__ == "__main__":
    main()