from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.analysis._base_analysis import base_analysis_pipeline

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Analysis flags to specify which analyses to run
analysis_flags = {
    "dataset_statistics": True,
    "language_distribution": True,
    "image_path_analysis": True,
    "data_anomalies": True,
    "conversation_tokens": True
}

# Run the base analysis
results = dataset.base_analysis_pipeline(analysis_flags=analysis_flags, output_dir="analysis_results")

# Print the results
print("Analysis results:", results)