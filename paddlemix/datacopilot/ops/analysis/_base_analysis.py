import os
from collections import Counter
import fasttext
import requests
from paddlenlp.transformers import AutoTokenizer
from paddlemix.datacopilot.core import MMDataset, register, ParallelMode
from typing import Dict, Any
from functools import partial
import json

from ..visualize._analysis_plot import visualize_results



FASTTEXT_MODEL_PATH = "lid.176.bin"
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def load_fasttext_model(model_path: str = FASTTEXT_MODEL_PATH, model_url: str = FASTTEXT_MODEL_URL) -> fasttext.FastText._FastText:
    """
    Check and load the FastText language detection model. If the model file does not exist locally, it will be downloaded.

    Args:
        model_path (str): Path to the FastText model file. Default is "lid.176.bin".
        model_url (str): URL to download the FastText model if it is not found locally.

    Returns:
        fasttext.FastText._FastText: Loaded FastText model instance.
    """
    # Check if the model already exists
    if not os.path.exists(model_path):
        print(f"FastText model file not found at {model_path}. Downloading...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()  # Raise an error for HTTP issues
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
                    f.write(chunk)
            print(f"FastText model successfully downloaded to {model_path}.")
        except Exception as e:
            print(f"Failed to download FastText model. Error: {e}")
            raise

    # Load the model
    try:
        print(f"Loading FastText model from {model_path}...")
        return fasttext.load_model(model_path)
    except Exception as e:
        print(f"Failed to load FastText model from {model_path}. Error: {e}")
        raise


def detect_language_with_fasttext(text: str, lang_model) -> str:
    """
    Detect the language of a given text using FastText.

    Args:
        text (str): Input text for language detection.
        lang_model: Loaded FastText model.

    Returns:
        str: Detected language code (e.g., 'en', 'fr'). Returns "unknown" if detection fails.
    """
    try:
        prediction = lang_model.predict(text.strip(), k=1)
        return prediction[0][0].replace("__label__", "")  # Return the language code
    except Exception:
        return "unknown"


def save_data_to_json(data: Any, filename: str):
    """
    Save data to a JSON file.

    Args:
        data (Any): Data to be saved.
        filename (str): Path to the JSON file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)




def compute_dataset_statistics(dataset: MMDataset) -> Dict[str, Any]:
    """
    Analyze the dataset and compute basic statistics, including valid and invalid items.

    Args:
        dataset (MMDataset): The dataset to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing dataset statistics.
    """
    # Initialize counters
    valid_items = []
    invalid_count = 0

    # Validate items in the dataset
    for item in dataset.items:
        # Check if the item has both "image" and valid "conversations"
        if "image" in item and isinstance(item.get("conversations"), list) and item["conversations"]:
            valid_items.append(item)
        else:
            invalid_count += 1

    # Count statistics for valid items
    conversation_counts = [
        len(item.get("conversations", [])) for item in valid_items
    ]
    total_conversations = sum(conversation_counts)  # Total number of Q&A pairs
    max_conversations = max(conversation_counts, default=0)  # Maximum Q&A pairs in a conversation
    min_conversations = min(conversation_counts, default=0)  # Minimum Q&A pairs in a conversation
    avg_conversations = total_conversations / len(conversation_counts) if conversation_counts else 0  # Average Q&A pairs

    # Count unique images
    unique_images = len(set(item.get("image", None) for item in valid_items if "image" in item))

    # Return the statistics
    return {
        "total_records": len(dataset),  # Total number of items in the dataset
        "unique_images": unique_images,  # Number of unique images
        "total_conversations": total_conversations,  # Total number of Q&A pairs
        "max_conversations": max_conversations,  # Maximum Q&A pairs in a conversation
        "min_conversations": min_conversations,  # Minimum Q&A pairs in a conversation
        "avg_conversations": avg_conversations,  # Average Q&A pairs per conversation
        "invalid_item_count": invalid_count,  # Number of invalid items
        "valid_items": valid_items,  # List of valid items
    }



def analyze_language_distribution(dataset: MMDataset, lang_model) -> Dict[str, Any]:
    """
    Analyze the language distribution in the dataset.

    Args:
        dataset (MMDataset): The dataset to analyze.
        lang_model: Loaded FastText model.

    Returns:
        Dict[str, Any]: Language distribution and mismatched language statistics.
    """
    human_msgs, assistant_msgs = [], []
    languages = Counter()
    mismatched_language_pairs = 0
    mismatched_pairs = []

    def process_conversation(item):
        nonlocal mismatched_language_pairs
        human_lang = assistant_lang = None

        for conv in item.get("conversations", []):
            if len(conv) < 2:
                continue

            human_text = conv[0]
            assistant_text = conv[1]

            human_msgs.append(human_text)
            assistant_msgs.append(assistant_text)

            human_lang = detect_language_with_fasttext(human_text, lang_model)
            assistant_lang = detect_language_with_fasttext(assistant_text, lang_model)

            if human_lang != "unknown" and assistant_lang != "unknown" and human_lang != assistant_lang:
                mismatched_language_pairs += 1
                mismatched_pairs.append({
                    "human_message": human_text,
                    "human_language": human_lang,
                    "assistant_message": assistant_text,
                    "assistant_language": assistant_lang
                })

            languages[human_lang] += 1
            languages[assistant_lang] += 1

    dataset.map(process_conversation, max_workers=8, mode=ParallelMode.THREAD, progress=True)

    return {
        "human_message_count": len(human_msgs),
        "assistant_message_count": len(assistant_msgs),
        "mismatched_language_pairs_count": mismatched_language_pairs,
        "languages_distribution": dict(languages),
    }


def validate_image_paths_in_dataset(dataset: MMDataset) -> Dict[str, Any]:
    """
    Validate the distribution and existence of image paths in the dataset.

    Args:
        dataset (MMDataset): The dataset to analyze.

    Returns:
        Dict[str, Any]: Image path statistics and missing path details.
    """
    def extract_image_path(item):
        return item.get("image", None)

    all_paths = dataset.map(extract_image_path, max_workers=8, mode=ParallelMode.THREAD, progress=True)
    all_paths = [path for path in all_paths if path]
    missing_paths = [path for path in all_paths if not os.path.exists(path)]
    path_distribution = Counter(os.path.dirname(path) for path in all_paths)

    return {
        "total_images": len(all_paths),
        "missing_images": len(missing_paths),
        "path_distribution": dict(path_distribution),
    }


def detect_data_anomalies(dataset: MMDataset, output_dir: str) -> Dict[str, int]:
    """
    Detect anomalies in the dataset.

    Args:
        dataset (MMDataset): The dataset to analyze.
        output_dir (str): Directory to save anomaly reports.

    Returns:
        Dict[str, int]: Counts of detected anomalies.
    """
    def identify_anomalies(item):
        anomalies = {}
        if not all(key in item for key in ["image", "conversations"]):
            anomalies["missing_fields"] = True
        if "conversations" in item and (not item["conversations"] or any(not conv[0].strip() for conv in item["conversations"])):
            anomalies["empty_conversations"] = True
        return anomalies

    anomaly_results = dataset.map(identify_anomalies, max_workers=8, mode=ParallelMode.THREAD)
    missing_fields = [item for item, result in zip(dataset, anomaly_results) if result.get("missing_fields")]
    empty_conversations = [item for item, result in zip(dataset, anomaly_results) if result.get("empty_conversations")]

    save_data_to_json(missing_fields, os.path.join(output_dir, "missing_fields.json"))
    save_data_to_json(empty_conversations, os.path.join(output_dir, "empty_conversations.json"))

    return {
        "missing_field_count": len(missing_fields),
        "empty_conversation_count": len(empty_conversations),
    }




def decode_token_ids(token_counts: Counter, tokenizer: AutoTokenizer) -> Counter:
    """
    Decode token IDs into their corresponding text and count their occurrences.

    Args:
        token_counts (Counter): A Counter object containing token IDs and their frequencies.
        tokenizer (AutoTokenizer): A tokenizer instance to decode the token IDs.

    Returns:
        Counter: A Counter object containing decoded tokens and their frequencies.
    """
    decoded_counts = Counter()
    for token_id, count in token_counts.items():
        try:
            decoded_text = tokenizer.decode([token_id]).strip()
            decoded_counts[decoded_text] += count
        except Exception as e:
            print(f"Error decoding token ID {token_id}: {e}")
    return decoded_counts


def analyze_conversation_tokens(item: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Analyze tokens in a single conversation.

    Args:
        item (Dict[str, Any]): A dictionary containing conversation data.
        tokenizer (AutoTokenizer): A tokenizer instance to tokenize the text.

    Returns:
        Dict[str, Any]: Token statistics for human and assistant messages.
    """
    human_tokens = []
    assistant_tokens = []

    for conv in item.get("conversations", []):
        try:
            if len(conv) > 0:  # Human message
                human_tokens.extend(tokenizer(conv[0], truncation=True, return_tensors="pd", use_fast=True)["input_ids"].numpy().flatten())
            if len(conv) > 1:  # Assistant message
                assistant_tokens.extend(tokenizer(conv[1], truncation=True, return_tensors="pd", use_fast=True)["input_ids"].numpy().flatten())
        except Exception as e:
            print(f"Error processing conversation: {conv}. Error: {e}")

    return {
        "human": {
            "total_tokens": len(human_tokens),
            "token_distribution": Counter(human_tokens),
        },
        "assistant": {
            "total_tokens": len(assistant_tokens),
            "token_distribution": Counter(assistant_tokens),
        }
    }


def run_token_analysis(dataset: MMDataset, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Perform token-level analysis on the dataset, including token distribution and frequency.

    Args:
        dataset (MMDataset): The dataset to analyze.
        tokenizer (AutoTokenizer): A tokenizer instance to tokenize the text in the dataset.

    Returns:
        Dict[str, Any]: Analysis results, including token distributions and high/low-frequency tokens.
    """
    print("Starting token analysis...")

    # Initialize counters and variables
    human_token_distribution = Counter()
    assistant_token_distribution = Counter()
    total_human_tokens = 0
    total_assistant_tokens = 0

    # Analyze tokens for all items in the dataset
    token_results = dataset.map(
        func=lambda item: analyze_conversation_tokens(item, tokenizer),
        max_workers=16,
        progress=True
    )

    # Aggregate results
    for result in token_results:
        total_human_tokens += result["human"]["total_tokens"]
        total_assistant_tokens += result["assistant"]["total_tokens"]
        human_token_distribution.update(result["human"]["token_distribution"])
        assistant_token_distribution.update(result["assistant"]["token_distribution"])



    # Identify high- and low-frequency tokens
    num_common_tokens = 20
    human_high_freq_tokens = human_token_distribution.most_common(num_common_tokens)
    assistant_high_freq_tokens = assistant_token_distribution.most_common(num_common_tokens)

    human_low_freq_tokens = human_token_distribution.most_common()[-num_common_tokens:]
    assistant_low_freq_tokens = assistant_token_distribution.most_common()[-num_common_tokens:]

    return {
        "human": {
            "total_tokens": total_human_tokens,
            "high_freq_tokens": decode_token_ids(Counter(dict(human_high_freq_tokens)), tokenizer),
            "low_freq_tokens": decode_token_ids(Counter(dict(human_low_freq_tokens)), tokenizer),
        },
        "assistant": {
            "total_tokens": total_assistant_tokens,
            "high_freq_tokens": decode_token_ids(Counter(dict(assistant_high_freq_tokens)), tokenizer),
            "low_freq_tokens": decode_token_ids(Counter(dict(assistant_low_freq_tokens)), tokenizer),
        }
    }
    
    

@register()
def run_analysis_pipeline(dataset: MMDataset, analysis_flags: Dict[str, bool] = None, output_dir: str = "output_directory") -> Dict[str, Any]:
    """
    Execute a pipeline of analysis functions on the dataset.

    Args:
        dataset (MMDataset): The dataset to analyze.
        analysis_flags (Dict[str, bool]): Flags to control which analyses to run.
        output_dir (str): Directory to save analysis results.

    Returns:
        Dict[str, Any]: Results of the analyses.
    """
    print("Initializing FastText model and Tokenizer...")
    lang_model = load_fasttext_model()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    if analysis_flags is None:
        analysis_flags = {
            "data_statistics": True,
            "field_distribution": True,
            "path_validation": True,
            "anomaly_detection": True,
            "token_analysis": True
        }

    results = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if analysis_flags.get("data_statistics", False):
        print("Running dataset statistics analysis...")
        results["data_statistics"] = compute_dataset_statistics(dataset)

    if analysis_flags.get("field_distribution", False):
        print("Running language distribution analysis...")
        results["field_distribution"] = analyze_language_distribution(dataset, lang_model)

    if analysis_flags.get("path_validation", False):
        print("Running image path validation...")
        results["path_validation"] = validate_image_paths_in_dataset(dataset)

    if analysis_flags.get("anomaly_detection", False):
        print("Running anomaly detection...")
        results["anomaly_detection"] = detect_data_anomalies(dataset, output_dir)

    if analysis_flags.get("token_analysis", False):
        print("Running token analysis...")
        results["token_analysis"] = run_token_analysis(dataset, tokenizer)

    print("All analyses completed. Visualizing results...")
    visualize_results(results, output_dir, analysis_flags)

    return results