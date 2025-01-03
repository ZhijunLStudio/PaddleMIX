from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.analysis._quality_analysis import quality_analysis

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Analysis flags to specify which analyses to run
quality_analysis_flags = {
    "image_text_matching": True,
    "object_detail_fulfillment": False,
    "caption_text_quality": False,
    "semantic_understanding": False,
}

# Apply the image caption metrics analysis operator
dataset_results = dataset.quality_analysis(
    model_name="Qwen/Qwen2.5-7B",  # Specify the model name
    quality_analysis_flags=quality_analysis_flags  # Pass the analysis flags
)

# Print the results of the evaluation
print("Evaluation results:", dataset_results)