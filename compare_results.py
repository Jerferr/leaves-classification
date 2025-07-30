import matplotlib.pyplot as plt
import numpy as np

# Model results data
models = ['VGG16', 'ViT', 'ResNet50']
accuracy = [0.97, 0.97, 1.00]
macro_f1 = [0.96, 0.96, 0.99]
weighted_f1 = [0.97, 0.97, 1.00]
error_count = [359, 403, 55]

x = np.arange(len(models))
width = 0.2

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot accuracy, macro F1, weighted F1
rects1 = ax1.bar(x - width, accuracy, width, label='Accuracy')
rects2 = ax1.bar(x, macro_f1, width, label='Macro F1')
rects3 = ax1.bar(x + width, weighted_f1, width, label='Weighted F1')

ax1.set_ylabel('Score')
ax1.set_ylim(0.9, 1.05)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_title('Model Accuracy and F1 Score Comparison')
ax1.legend(loc='upper left')

# Add error count bar chart on the right axis
ax2 = ax1.twinx()
rects4 = ax2.bar(x + 2*width, error_count, width, color='red', alpha=0.3, label='Error Count')
ax2.set_ylabel('Error Count')
ax2.set_ylim(0, max(error_count)*1.2)

# Add value labels
for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax1.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)
for rect in rects4:
    height = rect.get_height()
    ax2.annotate(f'{int(height)}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, color='red')

fig.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.savefig('model_comparison.png')
plt.show()
print('Comparison chart saved as model_comparison.png') 