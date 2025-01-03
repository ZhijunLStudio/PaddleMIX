from typing import Optional
from functools import partial
from ...core import MMDataset, register
import spacy

# python -m spacy download en_core_web_sm

# Load spaCy model
def load_spacy_model(lang: str):
    """
    Load the spaCy model based on the specified language.

    Args:
        lang (str): Language code, supports 'en' (English).

    Returns:
        spacy.Language: An instance of the spaCy language model.
    """
    if lang == 'en':
        return spacy.load("en_core_web_sm")  # English
    else:
        raise ValueError(f"Unsupported language: {lang}")


def is_action_count_valid(item, nlp, min_action_num: int = 1) -> bool:
    """
    Check if the number of verbs in the sample is greater than or equal to the specified minimum.

    Args:
        item (dict): A dictionary containing text information for the sample.
        nlp (spacy.Language): Loaded spaCy model.
        min_action_num (int): Minimum number of verbs. Default is 1.

    Returns:
        bool: True if the number of verbs is greater than or equal to min_action_num; otherwise, False.
    """
    # Get the text content and clean special characters
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # Process the text using the spaCy model
    doc = nlp(user_conv)

    # Define rules for detecting verbs based on language
    action_poss = ['VERB']
    action_tags = ['VB', 'VBP', 'VBZ', 'VBD', 'VBG', 'VBN']

    # Count the number of verbs in the text
    num_actions = sum(1 for token in doc if token.pos_ in action_poss and token.tag_ in action_tags)

    # Check if the verb count meets the requirement
    return num_actions >= min_action_num


@register()
def text_action_filter(
    dataset, 
    lang: str = 'en', 
    min_action_num: Optional[int] = 1
) -> MMDataset:
    """
    Filter the dataset based on the number of verbs in the samples.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        lang (str): Language of the text, supports 'en' (English).
        min_action_num (int): Minimum number of verbs. Default is 1.

    Returns:
        MMDataset: The filtered dataset.
    """
    print(f"Filtering samples based on language {lang} and minimum verb count {min_action_num}...")
    
    # Load the spaCy model (load once)
    nlp = load_spacy_model(lang)

    # Create the filter function
    filter_func = partial(is_action_count_valid, nlp=nlp, min_action_num=min_action_num)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset