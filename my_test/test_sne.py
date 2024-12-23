from paddlemix.datacopilot.core import MMDataset


# 加载数据集
dataset = MMDataset.from_json('./datasets/llava/02_val_chatml_filter.json')
print(f"原始数据集大小: {len(dataset)}")

# 执行LDA主题聚类和T-SNE可视化
lda_tsne_result = dataset.lda_topic_clustering(
    num_topics=2, 
    tsne_perplexity=5, 
    tsne_learning_rate=200, 
    tsne_n_iter=1000, 
    random_state=42, 
    output_plot="lda_tsne_visualization.png"
)

# 输出结果
lda_result = lda_tsne_result["lda_result"]
tsne_result = lda_tsne_result["tsne_result"]
topics = lda_tsne_result["topics"]
print("主题分布:", topics)
