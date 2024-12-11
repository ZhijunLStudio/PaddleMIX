import os
# os.environ["HF_DATASETS_DOWNLOAD_ATTEMPTS"] = "10"
# os.environ["HF_DATASETS_DOWNLOAD_TIMEOUT"] = "60"
# os.environ["HUGGINGFACE_HUB_URL"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-hub"
# os.environ["HF_DATASETS_URL"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-datasets"
# os.environ["HF_METRICS_URL"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-metrics"


from datasets import load_dataset

# 'Agriculture', 
choise = [
    'Accounting', 'Architecture_and_Engineering', 'Art', 
    'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 
    'Clinical_Medicine', 'Computer_Science', 'Design', 
    'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 
    'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 
    'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
]




# Loop through each subject in the choise list
for subject in choise:
    print(f"\nLoading dataset for: {subject}")
    
    try:
        # Load the dataset for the current subject
        dataset = load_dataset("MMMU/MMMU", subject, cache_dir="./cache")
        
        # Access the test split
        test_split = dataset['test']
        
        print(f"Processing {len(test_split)} samples for subject: {subject}")
        
        # Iterate through the test split
        for example in test_split:
            question = example['question']
            options = example['options']
            answer = example['answer']
            images = [example[f'image_{i+1}'] for i in range(7) if example[f'image_{i+1}']]  # Collect available images
            
            # Print details for each sample
            print("\nQuestion:", question)
            print("Options:", options)
            print("Answer:", answer)
            print("Images:", images)
    
    except Exception as e:
        print(f"Failed to load or process dataset for {subject}: {e}")


