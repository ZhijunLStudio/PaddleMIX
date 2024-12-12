from datasketch import MinHash, MinHashLSH
from simhash import Simhash
from typing import Dict
from tqdm import tqdm
from ...core import MMDataset, register
from ...misc import parallel_map, ParallelMode


def preprocess_text(text: str) -> str:
    """
    清理文本中的占位符 `<image>` 和多余换行符。
    
    参数:
        text (str): 原始文本。
    
    返回:
        str: 清理后的文本。
    """
    return text.replace("<image>", "").replace("\n", " ").strip()


def simhash_duplicate_operator(
    text: str,
    seen_hashes: set,
    threshold: float = 0.8
) -> bool:
    """
    使用 SimHash 算法检查文本是否重复。
    
    参数:
        text (str): 输入文本。
        seen_hashes (set): 已记录的 SimHash 值集合。
        threshold (float): 汉明距离的相似度阈值（默认 0.8）。
    
    返回:
        bool: 如果文本是重复的，返回 True，否则返回 False。
    """
    simhash_value = Simhash(text).value
    for existing_hash in seen_hashes:
        # 计算汉明距离
        distance = bin(simhash_value ^ existing_hash).count("1")
        if distance <= int((1 - threshold) * 64):  # SimHash 长度为 64
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
    使用 MinHashLSH 算法检查文本是否重复。
    
    参数:
        text (str): 输入文本。
        lsh (MinHashLSH): MinHashLSH 实例。
        threshold (float): Jaccard 相似度阈值（默认 0.8）。
        num_perm (int): MinHash 使用的哈希函数数量。
    
    返回:
        bool: 如果文本是重复的，返回 True，否则返回 False。
    """
    minhash = MinHash(num_perm=num_perm)
    for word in text.split():
        minhash.update(word.encode('utf8'))
    
    if list(lsh.query(minhash)):
        return True
    lsh.insert(len(lsh), minhash)
    return False



@register()
def remove_text_duplicates(
    dataset: MMDataset,
    method: str = "simhash",
    threshold: float = 0.8,
    num_perm: int = 128,
    print_duplicates: bool = False,
    max_workers: int = 8
) -> MMDataset:
    """
    基于 SimHash 或 MinHashLSH 对每个聊天中的每个问答对去重，同时记录重复关系。
    
    参数:
        dataset (MMDataset): 输入数据集。
        method (str): 使用的去重方法，可选 'simhash' 或 'minhash'。
        threshold (float): 相似度阈值（对 SimHash 表示汉明距离比例，对 MinHash 表示 Jaccard 相似度）。
        num_perm (int): MinHash 使用的哈希函数数量（仅用于 MinHash）。
        print_duplicates (bool): 是否打印重复关系，默认为 False。
    
    返回:
        MMDataset: 去重后的数据集。
    """
    if method not in {"simhash", "minhash"}:
        raise ValueError("Unsupported method. Choose 'simhash' or 'minhash'.")

    # 初始化计数器和重复关系记录
    total_pairs = 0
    removed_pairs = 0
    duplicate_relationships = []  # 用于存储重复关系 [(重复文本, 原始文本)]

    def filter_unique_conversations(item: Dict) -> Dict:
        """
        处理每条聊天记录中的问答对，逐对去重。
        """
        nonlocal total_pairs, removed_pairs
        local_seen_hashes = {}
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm) if method == "minhash" else None
        lsh_text_map = {}  # MinHash 查询结果与文本的映射
        unique_conversations = []

        for question, answer in item.get("conversations", []):
            text = f"{preprocess_text(question)} {preprocess_text(answer)}"
            total_pairs += 1  # 增加总计问答对计数
            if method == "simhash":
                simhash_value = Simhash(text).value
                found_duplicate = False
                for existing_text, existing_hash in local_seen_hashes.items():
                    distance = bin(simhash_value ^ existing_hash).count("1")
                    if distance <= int((1 - threshold) * 64):  # SimHash 长度为 64
                        duplicate_relationships.append((text, existing_text))
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
                    original_text = lsh_text_map[similar_items[0]]
                    duplicate_relationships.append((text, original_text))  # 记录第一个重复项
                    removed_pairs += 1
                else:
                    lsh.insert(len(lsh_text_map), minhash)
                    lsh_text_map[len(lsh_text_map)] = text
                    unique_conversations.append([question, answer])

        if unique_conversations:
            return {"image": item["image"], "conversations": unique_conversations}
        return None

    # 并发处理每条聊天记录
    filtered_items = parallel_map(
        filter_unique_conversations,
        dataset.items,
        max_workers=max_workers,
        mode=ParallelMode.THREAD,
        progress=True,
        order=False,
    )


    # 输出统计信息
    retained_pairs = total_pairs - removed_pairs
    print(f"总问答对数量: {total_pairs}")
    print(f"过滤掉的问答对数量: {removed_pairs}")
    print(f"最终保留的问答对数量: {retained_pairs}")

    if print_duplicates:
        print("\n重复关系:")
        for duplicate, original in duplicate_relationships[:20]:
            print(f"重复文本: {duplicate}")
            print(f"原始文本: {original}")
            print("---")

    return filtered_items