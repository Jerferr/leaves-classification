from collections import Counter
import os

def analyze_class_distribution(data_dir):
    class_counts = Counter()
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_name] = count
    print('类别分布:')
    for cls, cnt in class_counts.items():
        print(f'{cls}: {cnt}')
    return class_counts 