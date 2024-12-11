from modelscope.msdatasets import MsDataset


ds =  MsDataset.load('lmms-lab/MMMU')
# Access the test split
test_split = ds['dev']

# Iterate through the test split
for example in test_split:
    id = example['id']
    question = example['question']
    options = example['options']
    answer = example['answer']
    explanation = example['explanation']
    images = [example[f'image_{i+1}'] for i in range(7) if example[f'image_{i+1}']]  # Collect available images
    
    # Print details for each sample
    print("\nID:", id)
    print("\nQuestion:", question)
    print("Options:", options)
    print("explanation:", explanation)
    print("Answer:", answer)
    print("Images:", images)