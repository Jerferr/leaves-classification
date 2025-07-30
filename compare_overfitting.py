import subprocess
import sys
import time

def run_evaluation(model_name, weights_file, data_dir, description):
    """Run model evaluation and return results"""
    print(f"\n{'='*50}")
    print(f"Evaluating: {description}")
    print(f"Model: {model_name}, Weights: {weights_file}, Data: {data_dir}")
    print(f"{'='*50}")
    
    cmd = [
        'python3', 'eval.py',
        '--data_dir', data_dir,
        '--model', model_name,
        '--weights', weights_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("Evaluation completed!")
            return result.stdout
        else:
            print(f"Evaluation failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("Evaluation timeout!")
        return None
    except Exception as e:
        print(f"Runtime error: {e}")
        return None

def extract_metrics(output):
    """Extract key metrics from output"""
    if not output:
        return None
        
    lines = output.split('\n')
    metrics = {}
    
    for line in lines:
        if 'accuracy' in line and 'weighted avg' not in line and 'macro avg' not in line:
            try:
                accuracy = float(line.split()[-2])
                metrics['accuracy'] = accuracy
            except:
                pass
        elif 'macro avg' in line:
            try:
                parts = line.split()
                metrics['macro_f1'] = float(parts[-2])
            except:
                pass
        elif 'weighted avg' in line:
            try:
                parts = line.split()
                metrics['weighted_f1'] = float(parts[-2])
            except:
                pass
        elif 'Error samples count:' in line:
            try:
                error_count = int(line.split(':')[1].split(',')[0].strip())
                metrics['error_count'] = error_count
            except:
                pass
    
    return metrics

def main():
    print("Starting comparison of original and improved models for overfitting...")
    
    # Check if improved model weights exist
    import os
    if not os.path.exists('resnet_improved_best.pth'):
        print("\nWarning: Improved model weights file 'resnet_improved_best.pth' not found")
        print("Please run first: python3 train_improved.py --data_dir data/train --val_dir data/val")
        return
    
    results = {}
    
    # Evaluate original ResNet50
    print("\n" + "="*60)
    print("Phase 1: Evaluating Original ResNet50 Model")
    print("="*60)
    
    # Training set
    train_output = run_evaluation('resnet', 'resnet_best.pth', 'data/train', 
                                 'Original ResNet50 - Training Set')
    results['original_train'] = extract_metrics(train_output)
    
    # Validation set
    val_output = run_evaluation('resnet', 'resnet_best.pth', 'data/val', 
                               'Original ResNet50 - Validation Set')
    results['original_val'] = extract_metrics(val_output)
    
    # Evaluate improved ResNet50
    print("\n" + "="*60)
    print("Phase 2: Evaluating Improved ResNet50 Model")
    print("="*60)
    
    # Training set
    train_output_improved = run_evaluation('resnet_improved', 'resnet_improved_best.pth', 'data/train',
                                          'Improved ResNet50 - Training Set')
    results['improved_train'] = extract_metrics(train_output_improved)
    
    # Validation set
    val_output_improved = run_evaluation('resnet_improved', 'resnet_improved_best.pth', 'data/val',
                                        'Improved ResNet50 - Validation Set')
    results['improved_val'] = extract_metrics(val_output_improved)
    
    # Generate comparison report
    print("\n" + "="*80)
    print("Overfitting Comparison Analysis Report")
    print("="*80)
    
    if all(results.values()):
        print(f"{'Metric':<20} {'Original(Train)':<15} {'Original(Val)':<15} {'Improved(Train)':<15} {'Improved(Val)':<15}")
        print("-" * 80)
        
        # Accuracy
        print(f"{'Accuracy':<20} {results['original_train'].get('accuracy', 'N/A'):<15} "
              f"{results['original_val'].get('accuracy', 'N/A'):<15} "
              f"{results['improved_train'].get('accuracy', 'N/A'):<15} "
              f"{results['improved_val'].get('accuracy', 'N/A'):<15}")
        
        # Macro F1
        print(f"{'Macro F1':<20} {results['original_train'].get('macro_f1', 'N/A'):<15} "
              f"{results['original_val'].get('macro_f1', 'N/A'):<15} "
              f"{results['improved_train'].get('macro_f1', 'N/A'):<15} "
              f"{results['improved_val'].get('macro_f1', 'N/A'):<15}")
        
        # Error count
        print(f"{'Error Count':<20} {results['original_train'].get('error_count', 'N/A'):<15} "
              f"{results['original_val'].get('error_count', 'N/A'):<15} "
              f"{results['improved_train'].get('error_count', 'N/A'):<15} "
              f"{results['improved_val'].get('error_count', 'N/A'):<15}")
        
        # Overfitting analysis
        print("\nOverfitting Degree Analysis:")
        print("-" * 40)
        
        if (results['original_train'].get('error_count') and 
            results['original_val'].get('error_count')):
            original_gap = (results['original_val']['error_count'] - 
                           results['original_train']['error_count'])
            print(f"Original model error gap: {original_gap}")
        
        if (results['improved_train'].get('error_count') and 
            results['improved_val'].get('error_count')):
            improved_gap = (results['improved_val']['error_count'] - 
                           results['improved_train']['error_count'])
            print(f"Improved model error gap: {improved_gap}")
            
            if 'original_gap' in locals():
                if improved_gap < original_gap:
                    print("✅ Improved model shows reduced overfitting!")
                elif improved_gap > original_gap:
                    print("❌ Improved model shows increased overfitting")
                else:
                    print("➖ No significant change in overfitting degree")
    
    print("\nComparison completed! Please check the detailed results above.")

if __name__ == "__main__":
    main() 