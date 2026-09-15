"""
Student Feedback Sentiment Model - Evaluation Script
====================================================
Run this after training to complete:
- Test evaluation
- Generate plots
- Save results
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION (must match training!)
# ============================================================

CONFIG = {
    'data_path': 'Coursera_reviews.csv',
    'base_model': './distilbert-base-uncased',
    'output_dir': 'teacher_sentiment_model',
    'num_classes': 3,
    'class_names': ['Negative', 'Neutral', 'Positive'],
    'class_mapping': {
        0: 0,  # 1-star → Negative
        1: 0,  # 2-star → Negative
        2: 1,  # 3-star → Neutral
        3: 2,  # 4-star → Positive
        4: 2,  # 5-star → Positive
    },
    'max_length': 96,
    'batch_size': 128,
    'test_size': 0.1,
    'seed': 42,
    'num_workers': 4,
    'use_amp': True,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def tokenize_batch(texts, tokenizer, max_length, desc="Tokenizing"):
    all_input_ids = []
    all_attention_masks = []
    batch_size = 10000
    
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i:i+batch_size].tolist()
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        all_input_ids.append(encodings['input_ids'])
        all_attention_masks.append(encodings['attention_mask'])
    
    return torch.cat(all_input_ids, dim=0), torch.cat(all_attention_masks, dim=0)


def main():
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("STUDENT FEEDBACK SENTIMENT MODEL - EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    os.makedirs('plots', exist_ok=True)
    
    # ============================================================
    # FIX CONFIG.JSON (the bug from training)
    # ============================================================
    
    print("FIXING MODEL CONFIG")
    print("-" * 70)
    
    # Load original config from base model
    original_config = AutoConfig.from_pretrained(
        CONFIG['base_model'],
        local_files_only=True
    )
    
    # Update for our task
    original_config.num_labels = CONFIG['num_classes']
    original_config.id2label = {i: name for i, name in enumerate(CONFIG['class_names'])}
    original_config.label2id = {name: i for i, name in enumerate(CONFIG['class_names'])}
    
    # Save corrected config
    original_config.save_pretrained(CONFIG['output_dir'])
    print(f"  ✓ Fixed config.json in {CONFIG['output_dir']}/")
    
    # Save our training config separately
    training_config = {
        'num_classes': CONFIG['num_classes'],
        'class_names': CONFIG['class_names'],
        'class_mapping': CONFIG['class_mapping'],
        'max_length': CONFIG['max_length'],
    }
    with open(os.path.join(CONFIG['output_dir'], 'training_config.json'), 'w') as f:
        json.dump(training_config, f, indent=2)
    print(f"  ✓ Saved training_config.json")
    print()
    
    # ============================================================
    # LOAD DATA (only need test set)
    # ============================================================
    
    print("LOADING DATA")
    print("-" * 70)
    
    df = pd.read_csv(CONFIG['data_path'])
    print(f"Raw data: {len(df):,} samples")
    
    # Clean
    df = df.dropna(subset=['reviews', 'rating'])
    df = df[df['reviews'].str.strip() != '']
    df['rating'] = df['rating'].astype(int)
    df = df[df['rating'].between(1, 5)]
    
    # Map to 3 classes
    df['label_5class'] = df['rating'] - 1
    df['label'] = df['label_5class'].map(CONFIG['class_mapping'])
    
    print(f"Cleaned data: {len(df):,} samples")
    
    # Get test split (same as training!)
    _, X_test, _, y_test = train_test_split(
        df['reviews'].values, df['label'].values,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['seed'],
        stratify=df['label'].values
    )
    
    print(f"Test samples: {len(X_test):,}")
    print()
    
    del df
    gc.collect()
    
    # ============================================================
    # TOKENIZE TEST DATA
    # ============================================================
    
    print("TOKENIZATION")
    print("-" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['output_dir'], local_files_only=True)
    
    test_ids, test_masks = tokenize_batch(X_test, tokenizer, CONFIG['max_length'], "Test")
    test_labels = torch.tensor(y_test, dtype=torch.long)
    
    test_dataset = TensorDataset(test_ids, test_masks, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    print(f"Test batches: {len(test_loader):,}")
    print()
    
    del X_test, y_test
    gc.collect()
    
    # ============================================================
    # LOAD MODEL
    # ============================================================
    
    print("LOADING MODEL")
    print("-" * 70)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['output_dir'],
        local_files_only=True
    )
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded from {CONFIG['output_dir']}/")
    print(f"  ✓ Num labels: {model.config.num_labels}")
    print()
    
    # ============================================================
    # RUN TEST EVALUATION
    # ============================================================
    
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc="Testing", ncols=100):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            if CONFIG['use_amp']:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            probs = F.softmax(outputs.logits, dim=-1)
            _, preds = outputs.logits.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    test_acc = 100 * (all_preds == all_labels).mean()
    
    print()
    print(f"Test Accuracy: {test_acc:.2f}%")
    print()
    
    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================
    
    print("CLASSIFICATION REPORT")
    print("-" * 70)
    
    report = classification_report(
        all_labels, all_preds,
        target_names=CONFIG['class_names'],
        digits=3,
        output_dict=True
    )
    
    print(classification_report(
        all_labels, all_preds,
        target_names=CONFIG['class_names'],
        digits=3
    ))
    
    # ============================================================
    # TEACHER-FOCUSED METRICS
    # ============================================================
    
    print()
    print("=" * 70)
    print("📊 TEACHER-FOCUSED METRICS")
    print("=" * 70)
    print()
    
    # Negative class recall
    negative_recall = report['Negative']['recall'] * 100
    negative_precision = report['Negative']['precision'] * 100
    negative_f1 = report['Negative']['f1-score'] * 100
    
    print(f"  🔴 NEGATIVE FEEDBACK DETECTION (Struggling Students):")
    print(f"     Recall:    {negative_recall:.1f}% ← Catches {negative_recall:.0f}% of struggling students")
    print(f"     Precision: {negative_precision:.1f}% ← {negative_precision:.0f}% of flags are real issues")
    print(f"     F1-Score:  {negative_f1:.1f}%")
    print()
    
    # False negative analysis
    false_negatives = ((all_labels == 0) & (all_preds != 0)).sum()
    total_negatives = (all_labels == 0).sum()
    missed_pct = 100 * false_negatives / total_negatives if total_negatives > 0 else 0
    
    print(f"  ⚠️  MISSED STRUGGLING STUDENTS:")
    print(f"     {false_negatives:,} of {total_negatives:,} negative cases missed ({missed_pct:.1f}%)")
    print()
    
    # Where did false negatives go?
    fn_mask = (all_labels == 0) & (all_preds != 0)
    if fn_mask.sum() > 0:
        fn_preds = all_preds[fn_mask]
        fn_to_neutral = (fn_preds == 1).sum()
        fn_to_positive = (fn_preds == 2).sum()
        print(f"     Misclassified as Neutral:  {fn_to_neutral:,}")
        print(f"     Misclassified as Positive: {fn_to_positive:,}")
        print()
    
    # Confidence analysis
    pred_confidence = all_probs.max(axis=1)
    low_confidence = (pred_confidence < 0.7).sum()
    low_conf_pct = 100 * low_confidence / len(pred_confidence)
    
    print(f"  🤔 UNCERTAIN PREDICTIONS (confidence < 70%):")
    print(f"     {low_confidence:,} of {len(pred_confidence):,} predictions ({low_conf_pct:.1f}%)")
    print(f"     → These should be flagged for manual review")
    print()
    
    # Confidence by class
    print(f"  📈 AVERAGE CONFIDENCE BY PREDICTION:")
    for i, name in enumerate(CONFIG['class_names']):
        mask = all_preds == i
        if mask.sum() > 0:
            avg_conf = pred_confidence[mask].mean() * 100
            emoji = ['🔴', '🟡', '🟢'][i]
            print(f"     {emoji} {name}: {avg_conf:.1f}%")
    print()
    
    # ============================================================
    # CONFUSION MATRIX PLOT
    # ============================================================
    
    print("GENERATING PLOTS")
    print("-" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CONFIG['class_names'],
                yticklabels=CONFIG['class_names'], ax=axes[0],
                annot_kws={'size': 14})
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
    
    # Normalized (Recall)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=CONFIG['class_names'],
                yticklabels=CONFIG['class_names'], ax=axes[1],
                annot_kws={'size': 14})
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_title('Confusion Matrix (Recall per Class)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_3class.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: plots/confusion_matrix_3class.png")
    
    # ============================================================
    # PER-CLASS METRICS PLOT
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(3)
    width = 0.25
    
    recalls = [report[c]['recall'] * 100 for c in CONFIG['class_names']]
    precisions = [report[c]['precision'] * 100 for c in CONFIG['class_names']]
    f1s = [report[c]['f1-score'] * 100 for c in CONFIG['class_names']]
    
    bars1 = ax.bar(x - width, recalls, width, label='Recall', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x, precisions, width, label='Precision', color='#3498db', edgecolor='black')
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#2ecc71', edgecolor='black')
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Per-Class Metrics (Teacher Sentiment Model)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([
        '🔴 Negative\n(Needs Attention)',
        '🟡 Neutral\n(Mixed/Unclear)',
        '🟢 Positive\n(Satisfied)'
    ], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% target')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/per_class_metrics_3class.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: plots/per_class_metrics_3class.png")
    
    # ============================================================
    # CONFIDENCE DISTRIBUTION PLOT
    # ============================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall confidence distribution
    axes[0].hist(pred_confidence, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='70% threshold')
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Prediction Confidence Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confidence by class
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    for i, (name, color) in enumerate(zip(CONFIG['class_names'], colors)):
        mask = all_preds == i
        if mask.sum() > 0:
            axes[1].hist(pred_confidence[mask], bins=30, alpha=0.5, label=name, color=color)
    
    axes[1].axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='70% threshold')
    axes[1].set_xlabel('Confidence', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Confidence by Predicted Class', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/confidence_distribution.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: plots/confidence_distribution.png")
    
    # ============================================================
    # ERROR ANALYSIS PLOT
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate error rates
    error_rates = []
    for i, name in enumerate(CONFIG['class_names']):
        mask = all_labels == i
        errors = (all_preds[mask] != all_labels[mask]).sum()
        total = mask.sum()
        error_rate = 100 * errors / total if total > 0 else 0
        error_rates.append(error_rate)
    
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    bars = ax.bar(CONFIG['class_names'], error_rates, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Error Rate (%)', fontsize=12)
    ax.set_title('Error Rate by True Class', fontsize=14)
    ax.set_ylim(0, max(error_rates) * 1.2 if max(error_rates) > 0 else 10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, error_rates):
        ax.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/error_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: plots/error_analysis.png")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    
    print()
    print("SAVING RESULTS")
    print("-" * 70)
    
    results = {
        'test_accuracy': float(test_acc),
        'negative_recall': float(negative_recall),
        'negative_precision': float(negative_precision),
        'negative_f1': float(negative_f1),
        'neutral_recall': float(report['Neutral']['recall'] * 100),
        'positive_recall': float(report['Positive']['recall'] * 100),
        'missed_struggling_students': int(false_negatives),
        'total_negative_cases': int(total_negatives),
        'missed_percentage': float(missed_pct),
        'low_confidence_predictions': int(low_confidence),
        'low_confidence_percentage': float(low_conf_pct),
        'macro_f1': float(report['macro avg']['f1-score'] * 100),
        'weighted_f1': float(report['weighted avg']['f1-score'] * 100),
    }
    
    # Save as JSON
    with open(os.path.join(CONFIG['output_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: {CONFIG['output_dir']}/results.json")
    
    # Save full results as PyTorch
    full_results = {
        **results,
        'config': CONFIG,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'all_predictions': all_preds.tolist(),
        'all_labels': all_labels.tolist(),
    }
    torch.save(full_results, os.path.join(CONFIG['output_dir'], 'results.pt'))
    print(f"  ✓ Saved: {CONFIG['output_dir']}/results.pt")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print()
    print("=" * 70)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 70)
    print()
    print("  RESULTS SUMMARY:")
    print(f"    Test Accuracy:      {test_acc:.2f}%")
    print(f"    Macro F1-Score:     {report['macro avg']['f1-score']*100:.2f}%")
    print(f"    Weighted F1-Score:  {report['weighted avg']['f1-score']*100:.2f}%")
    print()
    print("  PER-CLASS RECALL (most important for teachers):")
    print(f"    🔴 Negative:  {negative_recall:.1f}% ← Catches {100-missed_pct:.0f}% of struggling students")
    print(f"    🟡 Neutral:   {report['Neutral']['recall']*100:.1f}%")
    print(f"    🟢 Positive:  {report['Positive']['recall']*100:.1f}%")
    print()
    print("  KEY INSIGHTS:")
    print(f"    • {false_negatives:,} struggling students would be missed ({missed_pct:.1f}%)")
    print(f"    • {low_confidence:,} predictions need manual review ({low_conf_pct:.1f}%)")
    print()
    print("  FILES SAVED:")
    print(f"    • {CONFIG['output_dir']}/results.json")
    print(f"    • {CONFIG['output_dir']}/results.pt")
    print(f"    • plots/confusion_matrix_3class.png")
    print(f"    • plots/per_class_metrics_3class.png")
    print(f"    • plots/confidence_distribution.png")
    print(f"    • plots/error_analysis.png")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()