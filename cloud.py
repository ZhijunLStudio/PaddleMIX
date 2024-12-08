from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 假设 `tokens` 是包含所有低频token的列表
tokens = ['eselect', '纺织', '烘干', '👙', '😥', '😓', 'cuz', '👻']  # 示例数据

# 设置正确的字体路径
font_path = "/home/lizhijun/PaddleMIX-develop/NotoColorEmoji.ttf"  # 修改为正确的字体路径

# 创建词云，指定字体路径
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                      font_path=font_path, max_font_size=10).generate(' '.join(tokens))  # 设置max_font_size

# 显示词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# 保存词云为图像
plt.savefig('wordcloud.png', format='png', bbox_inches='tight')  # 保存为 PNG 文件
