from typing import Optional, List, Dict
from datasketch import MinHash, MinHashLSH
from simhash import Simhash
from paddlemix.datacopilot.core import MMDataset, register

def compute_simhash(text: str) -> int:
    """计算文本的 SimHash 值，返回整数值"""
    return Simhash(text).value  # 直接返回整数值

def compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    """计算文本的 MinHash 值。"""
    minhash = MinHash(num_perm=num_perm)
    for word in text.split():
        minhash.update(word.encode('utf8'))
    return minhash

def extract_conversation_texts(conversations: List[Dict]) -> List[str]:
    """
    从对话中提取文本对
    每个文本对由连续的human和assistant消息组成
    """
    texts = []
    for i in range(0, len(conversations)-1, 2):
        if (conversations[i]['from'] == 'human' and 
            conversations[i+1]['from'] == 'assistant'):
            text = conversations[i]['value'].strip() + ' ' + conversations[i+1]['value'].strip()
            texts.append(text)
    return texts

@register()
def remove_text_duplicates(
    dataset: MMDataset,
    method: str = "simhash",
    threshold: float = 0.8,
    merge_text: bool = False,
    num_perm: int = 128
) -> MMDataset:
    """基于 SimHash 或 MinHashLSH 去除文本级别的重复样本"""
    filtered_items = []
    hash_dict = {}
    
    if method == "simhash":
        for idx, item in enumerate(dataset):
            # 提取所有对话文本对
            texts = extract_conversation_texts(item['conversations'])
            
            for text in texts:
                if not text:
                    continue
                
                # 直接使用整数值作为哈希键
                simhash_value = compute_simhash(text)
                found_similar = False
                
                for existing_hash, existing_data in hash_dict.items():
                    # 计算汉明距离
                    distance = bin(simhash_value ^ existing_hash).count('1')
                    if distance <= int((1 - threshold) * 64):  # 64 是 SimHash 的长度
                        found_similar = True
                        if merge_text:
                            existing_data['items'].append(item)
                        break
                
                if not found_similar:
                    hash_dict[simhash_value] = {
                        'items': [item],
                        'texts': [text]
                    }
    
    elif method == "minhash":
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        
        for idx, item in enumerate(dataset):
            texts = extract_conversation_texts(item['conversations'])
            
            for text in texts:
                if not text:
                    continue
                
                minhash = compute_minhash(text, num_perm)
                similar_items = list(lsh.query(minhash))
                
                if similar_items:
                    if merge_text:
                        for sim_idx in similar_items:
                            hash_dict[sim_idx]['items'].append(item)
                else:
                    lsh.insert(idx, minhash)
                    hash_dict[idx] = {
                        'items': [item],
                        'texts': [text]
                    }
    
    # 去重并保留第一个相似项
    unique_items = {}
    for data in hash_dict.values():
        representative_item = data['items'][0]
        unique_items[representative_item['id']] = representative_item
    
    # 将唯一项转换为列表
    filtered_items = list(unique_items.values())
    
    return MMDataset(filtered_items)