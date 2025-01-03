from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddlemix.datacopilot.core import MMDataset, register
from tqdm import tqdm
from typing import Dict, List

# Predefined evaluation metrics and corresponding prompt templates
CRITERIA_PROMPTS = {
    "image_text_matching": """Please evaluate if the provided text caption accurately represents the main features and objects of the image. The caption doesn't need to detail every aspect of the image, but it should capture its primary theme. Rate the overall quality of the text caption's match to the image on a scale of 1-100, considering the criteria mentioned.""",
    "object_detail_fulfillment": """Please evaluate the text caption to determine if it provides detailed descriptions of objects that align with the image. Specifically, assess if the caption sufficiently describes the color, size, position, shape, material, etc., of the objects. Afterward, rate the caption's overall accuracy in capturing object details from the image on a scale of 1-100, based on the criteria provided.""",
    "caption_text_quality": """Please evaluate the text caption based on the following criteria: Grammatical Correctness, Diversity of Vocabulary (e.g., the range and uniqueness of words used), Fluency (e.g., smoothness and natural flow of sentences), Readability, Length, and Structure. Assign an overall quality score on a scale of 1-100.""",
    "semantic_understanding": """Evaluate the given text caption in relation to its corresponding image. Your goal is to determine if the text caption provides additional semantic information that isn't readily apparent just from the image itself. Rate the text caption's semantic depth on a scale from 1 to 100.""",
}

DEFAULT_PROMPT_TEMPLATE = """Text Caption: {caption}

{criteria}
A higher score indicates a higher level of {aspect}. Ensure that your scoring is nuanced and uses the entire range from 0 to 100, reflecting the subtle differences. The score should be given as an integer, with each number between 0 and 100 considered as a potential score, avoiding the tendency to round to multiples of 10. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""

# Load the model and tokenizer
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
    return tokenizer, model

def evaluate_image_caption(
    dataset: MMDataset, 
    model_name: str = "Qwen/Qwen2.5-7B", 
    analysis_flags: Dict[str, bool] = None
) -> Dict:
    """
    Evaluate the quality of image captions based on predefined metrics.

    Args:
        dataset (MMDataset): The dataset containing image paths and conversations.
        model_name (str): Name of the model to use.
        analysis_flags (Dict[str, bool]): Flags to control which metrics to evaluate.

    Returns:
        Dict: Evaluation results for each dataset item.
    """
    # Load the model
    tokenizer, model = load_model(model_name)

    # Final results storage
    results = {}

    # Determine which metrics to evaluate based on analysis_flags
    if analysis_flags is None:
        selected_metrics = list(CRITERIA_PROMPTS.keys())  # Default: All metrics
    else:
        selected_metrics = [key for key, value in analysis_flags.items() if value]

    # Process in batches, each batch handles a set of data
    batch_size = 1
    batch_data = []

    for item in tqdm(dataset):
        item_id = item["image"]  # Use image path as item_id
        conversations = item["conversations"]
        
        # Combine all Q&A pairs into a single conversation
        full_caption = ""
        for conversation in conversations:
            question, answer = conversation
            question = question.replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
            full_caption += f"Question: {question}\nAnswer: {answer}\n"

        # Evaluate each selected metric
        for metric in selected_metrics:
            criteria = CRITERIA_PROMPTS[metric]
            aspect = metric.replace("_", " ")
            caption = full_caption

            # Generate the full prompt
            full_prompt = DEFAULT_PROMPT_TEMPLATE.format(
                caption=caption, 
                criteria=criteria, 
                aspect=aspect
            )
            
            # Store the generated prompt in the current batch
            batch_data.append(full_prompt)

            # Perform inference when batch size is reached
            if len(batch_data) == batch_size:
                # Tokenize input
                input_features = tokenizer(batch_data, return_tensors="pd", padding=True)

                # Model inference
                outputs = model.generate(**input_features, max_length=256)

                # Decode the output
                decoded_outputs = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)

                # Store results
                for idx, decoded_output in enumerate(decoded_outputs):
                    if item_id not in results:
                        results[item_id] = {}
                    results[item_id][metric] = decoded_output

                # Clear the current batch
                batch_data = []

    return results

@register()
def quality_analysis(dataset: MMDataset, model_name: str, quality_analysis_flags: Dict[str, bool] = None):
    """
    Analyze the quality of multi-turn conversations for image captioning.

    Args:
        dataset (MMDataset): The dataset containing image paths and conversations.
        model_name (str): Name of the model to use.
        analysis_flags (Dict[str, bool]): Flags to control which metrics to evaluate.

    Returns:
        Dict: Evaluation results for each dataset item.
    """
    results = evaluate_image_caption(dataset, model_name, quality_analysis_flags)
    return results