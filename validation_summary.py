#!/usr/bin/env python3
"""
Validation Summary Script
Summarizes the performance of all trained models
"""

import os
from datetime import datetime

def main():
    print("="*60)
    print("PLANT DISEASE CLASSIFICATION - MODEL VALIDATION SUMMARY")
    print("="*60)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check available model weights
    available_models = []
    model_files = {
        'VGG16': 'vgg16_best.pth',
        'ResNet50 (Original)': 'resnet_best.pth', 
        'ResNet50 (Improved)': 'resnet_improved_best.pth',
        'ResNet50 (Final)': 'resnet_improved_final.pth',
        'ViT': 'vit_best.pth'
    }
    
    print("AVAILABLE TRAINED MODELS:")
    print("-" * 30)
    for model_name, filename in model_files.items():
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024*1024)  # MB
            available_models.append((model_name, filename, file_size))
            print(f"✅ {model_name:<20} | {filename:<25} | {file_size:.1f} MB")
        else:
            print(f"❌ {model_name:<20} | {filename:<25} | Not found")
    
    print()
    print("VALIDATION RESULTS SUMMARY:")
    print("-" * 40)
    
    # Based on previous validation results
    results = {
        'VGG16': {
            'accuracy': 0.97,
            'macro_f1': 0.96,
            'weighted_f1': 0.97,
            'error_count': 359,
            'total_samples': 12307,
            'notes': 'Baseline performance, higher error count'
        },
        'ViT': {
            'accuracy': 0.97,
            'macro_f1': 0.96,
            'weighted_f1': 0.97,
            'error_count': 403,
            'total_samples': 12307,
            'notes': 'Similar to VGG16 but with more errors'
        },
        'ResNet50 (Original)': {
            'accuracy': 1.00,
            'macro_f1': 0.99,
            'weighted_f1': 1.00,
            'error_count': 55,
            'total_samples': 12307,
            'notes': 'Excellent performance, minimal errors'
        },
        'ResNet50 (Improved)': {
            'accuracy': 1.00,
            'macro_f1': 1.00,
            'weighted_f1': 1.00,
            'error_count': 46,
            'total_samples': 12307,
            'notes': 'Best performance with regularization'
        }
    }
    
    # Print results table
    print(f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'W. F1':<8} {'Errors':<8} {'Error %':<8}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        error_rate = (metrics['error_count'] / metrics['total_samples']) * 100
        print(f"{model_name:<20} {metrics['accuracy']:<10.2f} {metrics['macro_f1']:<10.2f} "
              f"{metrics['weighted_f1']:<8.2f} {metrics['error_count']:<8} {error_rate:<8.2f}%")
    
    print()
    print("PERFORMANCE RANKING:")
    print("-" * 20)
    
    # Sort by error count (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['error_count'])
    
    for i, (model_name, metrics) in enumerate(sorted_results, 1):
        print(f"{i}. {model_name}")
        print(f"   Accuracy: {metrics['accuracy']:.2f}, Errors: {metrics['error_count']}")
        print(f"   Notes: {metrics['notes']}")
        print()
    
    print("RECOMMENDATIONS:")
    print("-" * 16)
    print("🏆 PRODUCTION USE: ResNet50 (Improved) - Best overall performance")
    print("🔬 RESEARCH USE: All models available for comparison studies")
    print("📊 BASELINE: VGG16 - Good reference point for improvements")
    
    print()
    print("OVERFITTING ANALYSIS:")
    print("-" * 21)
    print("• Original ResNet50: Slight overfitting (training: 25 errors, val: 55 errors)")
    print("• Improved ResNet50: Reduced overfitting (training: ~20 errors, val: 46 errors)")
    print("• Regularization techniques successfully reduced overfitting")
    
    print()
    print("FILES GENERATED:")
    print("-" * 15)
    output_files = [
        'confusion_matrix.png',
        'error_samples.txt', 
        'error_samples/',
        'model_comparison.png',
        'model_comparison_report.txt'
    ]
    
    for filename in output_files:
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} (not found)")
    
    print()
    print("="*60)
    print("VALIDATION SUMMARY COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 