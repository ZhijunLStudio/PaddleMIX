from datasketch import MinHash, MinHashLSH
from simhash import Simhash
from typing import Dict
from tqdm import tqdm
from ...core import MMDataset, register
from ...misc import parallel_map, ParallelMode


def preprocess_text(text: str) -> str:
    """
    Cleans placeholder `<image>` and extra newlines from the text.

    Args:
        text (str): Original text.

    Returns:
        str: Cleaned text.
    """
    return text.replace("<image>", "").replace("\n<image>", " ").replace("<image>\n", " ").strip()


def simhash_duplicate_operator(
    text: str,
    seen_hashes: set,
    threshold: float = 0.8
) -> bool:
    """
    Checks for duplicate text using the SimHash algorithm.

    Args:
        text (str): Input text.
        seen_hashes (set): Set of previously recorded SimHash values.
        threshold (float): Similarity threshold for Hamming distance (default: 0.8).

    Returns:
        bool: True if the text is a duplicate, False otherwise.
    """
    simhash_value = Simhash(text).value
    for existing_hash in seen_hashes:
        # Calculate Hamming distance
        distance = bin(simhash_value ^ existing_hash).count("1")
        if distance <= int((1 - threshold) * 64):  # SimHash length is 64
            return True
    seen_hashes.add(simhash_value)
    return False


def minhash_duplicate_operator(
    text: str,
    lsh: MinHashLSH,
    threshold: float = 0.8,
    num_perm: int = 128
) -> bool:
    """
    Checks for duplicate text using the MinHashLSH algorithm.

    Args:
        text (str): Input text.
        lsh (MinHashLSH): MinHashLSH instance.
        threshold (float): Jaccard similarity threshold (default: 0.8).
        num_perm (int): Number of hash functions for MinHash.

    Returns:
        bool: True if the text is a duplicate, False otherwise.
    """
    minhash = MinHash(num_perm=num_perm)
    for word in text.split():
        minhash.update(word.encode('utf8'))
    
    if list(lsh.query(minhash)):
        return True
    lsh.insert(len(lsh), minhash)
    return False


@register()
def conversation_hash_filter(
    dataset: MMDataset,
    method: str = "simhash",
    threshold: float = 0.8,
    num_perm: int = 128,
    max_workers: int = 8
) -> MMDataset:
    """
    Removes duplicate Q&A pairs in conversations using either SimHash or MinHashLSH.

    Args:
        dataset (MMDataset): Input dataset.
        method (str): Deduplication method, either 'simhash' or 'minhash'.
        threshold (float): Similarity threshold (for SimHash, it is Hamming distance ratio; for MinHash, it is Jaccard similarity).
        num_perm (int): Number of hash functions for MinHash (only used for MinHash).

    Returns:
        MMDataset: Dataset after deduplication.
    """
    if method not in {"simhash", "minhash"}:
        raise ValueError("Unsupported method. Choose 'simhash' or 'minhash'.")

    # Initialize counters
    total_pairs = 0
    removed_pairs = 0

    def filter_unique_conversations(item: Dict) -> Dict:
        """
        Processes each conversation and removes duplicate Q&A pairs.
        """
        nonlocal total_pairs, removed_pairs
        local_seen_hashes = {}
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm) if method == "minhash" else None
        lsh_text_map = {}
        unique_conversations = []

        for question, answer in item.get("conversations", []):
            text = f"{preprocess_text(question)} {preprocess_text(answer)}"
            total_pairs += 1
            if method == "simhash":
                simhash_value = Simhash(text).value
                found_duplicate = False
                for existing_hash in local_seen_hashes.values():
                    distance = bin(simhash_value ^ existing_hash).count("1")
                    if distance <= int((1 - threshold) * 64):  # SimHash length is 64
                        removed_pairs += 1
                        found_duplicate = True
                        break
                if not found_duplicate:
                    local_seen_hashes[text] = simhash_value
                    unique_conversations.append([question, answer])
            elif method == "minhash":
                minhash = MinHash(num_perm=num_perm)
                for word in text.split():
                    minhash.update(word.encode('utf8'))
                similar_items = list(lsh.query(minhash))
                if similar_items:
                    removed_pairs += 1
                else:
                    lsh.insert(len(lsh_text_map), minhash)
                    lsh_text_map[len(lsh_text_map)] = text
                    unique_conversations.append([question, answer])

        if unique_conversations:
            return {"image": item["image"], "conversations": unique_conversations}
        return None

    # Process conversations in parallel
    filtered_items = parallel_map(
        filter_unique_conversations,
        dataset.items,
        max_workers=max_workers,
        mode=ParallelMode.THREAD,
        progress=True,
        order=False,
    )

    # Output statistics
    retained_pairs = total_pairs - removed_pairs
    print(f"Total Q&A pairs: {total_pairs}")
    print(f"Filtered Q&A pairs: {removed_pairs}")
    print(f"Remaining Q&A pairs: {retained_pairs}")

    return MMDataset(filtered_items)