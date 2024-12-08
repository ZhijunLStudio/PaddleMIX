import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from ...core import T, MMDataset, register

@register()
def extract_text_for_lda(item: T) -> Optional[str]:
    """从对话中提取文本，用于主题建模."""
    conversations = item.get("conversations", [])
    text = []
    for convo in conversations:
        text.append(convo["value"])
    return " ".join(text)

@register()
def lda_topic_clustering(
    dataset: MMDataset,
    num_topics: int = 5,
    tsne_perplexity: int = 30,
    tsne_learning_rate: int = 200,
    tsne_n_iter: int = 1000,
    random_state: int = 42,
    output_plot: str = "lda_tsne_plot.png"
):
    """对对话文本进行LDA主题聚类，并使用T-SNE进行可视化."""
    # 提取文本数据
    texts = dataset.map(extract_text_for_lda)
    texts = [text for text in texts if text.strip()]  # 移除空文本

    # 文本向量化
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    text_matrix = vectorizer.fit_transform(texts)

    # LDA 主题建模
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=random_state)
    lda_result = lda.fit_transform(text_matrix)

    # 使用 T-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate=tsne_learning_rate,
        n_iter=tsne_n_iter,
        random_state=random_state
    )
    tsne_result = tsne.fit_transform(lda_result)

    # 可视化结果
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        tsne_result[:, 0], tsne_result[:, 1], c=np.argmax(lda_result, axis=1), cmap='tab10', alpha=0.7
    )
    plt.colorbar(scatter, label="Topic Cluster")
    plt.title("LDA Topic Clustering with T-SNE Visualization")
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")
    plt.savefig(output_plot)
    plt.show()

    return {
        "lda_result": lda_result,
        "tsne_result": tsne_result,
        "topics": np.argmax(lda_result, axis=1).tolist()
    }
