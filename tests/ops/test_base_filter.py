from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._base_filter import valid_data_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the filter operator
dataset = dataset.valid_data_filter()

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Dataset validation complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_base_filter.json'))