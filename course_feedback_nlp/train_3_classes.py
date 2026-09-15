"""
Student Feedback Sentiment Model - Training Script
==================================================
Optimized for Teacher/Agent use case:
- 3 classes: Negative, Neutral, Positive
- High recall on negative feedback (don't miss struggling students)
- Confidence scores for uncertainty
- Fast inference for agent integration

FIXED: save_checkpoint now properly preserves model config.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import gc
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# AMD CRASH PROTECTION
# ============================================================
os.environ['AMD_LOG_LEVEL'] = '0'
os.environ['ROCM_LOG_LEVEL'] = '0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

# ============================================================
# CONFIGURATION - OPTIMIZED FOR TEACHER USE CASE
# ============================================================

CONFIG = {
    # ==================== DATA ====================
    'data_path': 'Coursera_reviews.csv',
    'model_name': './distilbert-base-uncased',
    'output_dir': 'teacher_sentiment_model',
    'checkpoint_dir': 'checkpoints_teacher',
    
    # ==================== CLASS MAPPING ====================
    # Map 5-star ratings to 3 classes
    'num_classes': 3,
    'class_names': ['Negative', 'Neutral', 'Positive'],
    'class_mapping': {
        0: 0,  # 1-star → Negative (0)
        1: 0,  # 2-star → Negative (0)
        2: 1,  # 3-star → Neutral (1)
        3: 2,  # 4-star → Positive (2)
        4: 2,  # 5-star → Positive (2)
    },
    
    # ==================== TOKENIZATION ====================
    'max_length': 96,
    
    # ==================== TRAINING ====================
    'batch_size': 128,
    'gradient_accumulation_steps': 2,
    'epochs': 7,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.06,
    'max_grad_norm': 1.0,
    
    # ==================== SCHEDULER ====================
    'scheduler_type': 'cosine',
    'cosine_min_lr_ratio': 0.01,
    
    # ==================== LOSS FUNCTION ====================
    'loss_type': 'focal',  # Focal loss to focus on hard examples
    'focal_gamma': 2.0,
    'label_smoothing': 0.05,  # Light smoothing for calibration
    
    # ==================== CLASS IMBALANCE ====================
    # IMPORTANT: Weight negative class higher - we don't want to miss struggling students!
    'use_class_weights': True,
    'class_weight_power': 0.7,  # Moderate-high weighting for minorities
    'negative_class_boost': 1.5,  # Extra boost for negative class (teacher priority)
    
    # ==================== EARLY STOPPING ====================
    'early_stopping': True,
    'early_stopping_patience': 3,
    'early_stopping_metric': 'val_loss',
    
    # ==================== HARDWARE ====================
    'seed': 42,
    'num_workers': 4,
    'pin_memory': True,
    'use_amp': True,
    
    # ==================== CHECKPOINTING ====================
    'checkpoint_every_epoch': True,
    'save_total_limit': 3,
    
    # ==================== DATA SPLIT ====================
    'train_size': 0.8,
    'val_size': 0.1,
    'test_size': 0.1,
}


# ============================================================
# CUSTOM LOSS FUNCTIONS
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing.
    Focuses training on hard-to-classify examples.
    """
    def __init__(self, num_classes=3, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Cross entropy with optional label smoothing
        if self.label_smoothing > 0:
            confidence = 1.0 - self.label_smoothing
            smooth_value = self.label_smoothing / (self.num_classes - 1)
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            smooth_targets = one_hot * confidence + (1 - one_hot) * smooth_value
            log_probs = F.log_softmax(logits, dim=-1)
            ce = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce = F.cross_entropy(logits, targets, reduction='none')
        
        loss = focal_weight * ce
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        return loss.mean()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_scheduler(optimizer, scheduler_type, total_steps, warmup_steps, config):
    if scheduler_type == 'cosine':
        min_lr_ratio = config.get('cosine_min_lr_ratio', 0.01)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, epoch,
                    val_acc, val_loss, history, config, path, is_best=False):
    """
    Save training checkpoint.
    
    FIXED: Now saves training_config.json separately instead of overwriting
    the model's config.json (which needs model_type for loading).
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'val_accuracy': val_acc,
        'val_loss': val_loss,
        'history': history,
        'config': config,
    }
    torch.save(checkpoint, path)
    
    if is_best:
        # Save model and tokenizer - this creates the correct config.json with model_type
        model.save_pretrained(config['output_dir'])
        tokenizer.save_pretrained(config['output_dir'])
        
        # FIXED: Save our custom training config to a SEPARATE file
        # DO NOT overwrite the model's config.json!
        training_config_path = os.path.join(config['output_dir'], 'training_config.json')
        training_config = {
            'num_classes': config['num_classes'],
            'class_names': config['class_names'],
            'class_mapping': {str(k): v for k, v in config['class_mapping'].items()},  # JSON needs string keys
            'max_length': config['max_length'],
        }
        with open(training_config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        
        # Also update the model's config.json with our label mappings (properly!)
        model_config = AutoConfig.from_pretrained(config['output_dir'])
        model_config.num_labels = config['num_classes']
        model_config.id2label = {i: name for i, name in enumerate(config['class_names'])}
        model_config.label2id = {name: i for i, name in enumerate(config['class_names'])}
        model_config.save_pretrained(config['output_dir'])


def cleanup_old_checkpoints(checkpoint_dir, save_total_limit):
    if save_total_limit is None or save_total_limit <= 0:
        return
    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
    ])
    while len(checkpoints) > save_total_limit:
        oldest = checkpoints.pop(0)
        os.remove(os.path.join(checkpoint_dir, oldest))


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


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    set_seed(CONFIG['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("STUDENT FEEDBACK SENTIMENT MODEL")
    print("Optimized for Teacher/Agent Use Case")
    print("=" * 70)
    print()
    print("TARGET CLASSES:")
    print("  🔴 Negative (1-2 stars) → 'Needs Attention'")
    print("  🟡 Neutral  (3 stars)   → 'Mixed/Unclear'")
    print("  🟢 Positive (4-5 stars) → 'Satisfied'")
    print()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    print()
    
    # Create directories
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # ============================================================
    # DATA LOADING & PREPROCESSING
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
    
    # Original 5-class labels
    df['label_5class'] = df['rating'] - 1
    
    # Map to 3 classes
    df['label'] = df['label_5class'].map(CONFIG['class_mapping'])
    
    print(f"Cleaned data: {len(df):,} samples")
    print()
    
    # Show original distribution
    print("Original 5-class distribution:")
    for rating in range(1, 6):
        count = (df['rating'] == rating).sum()
        pct = 100 * count / len(df)
        print(f"  {rating} Star: {count:>8,} ({pct:>5.1f}%)")
    print()
    
    # Show new 3-class distribution
    print("New 3-class distribution:")
    class_counts_3 = []
    for label, name in enumerate(CONFIG['class_names']):
        count = (df['label'] == label).sum()
        pct = 100 * count / len(df)
        class_counts_3.append(count)
        emoji = ['🔴', '🟡', '🟢'][label]
        print(f"  {emoji} {name}: {count:>8,} ({pct:>5.1f}%)")
    print()
    
    # ============================================================
    # CALCULATE CLASS WEIGHTS
    # ============================================================
    
    if CONFIG['use_class_weights']:
        print("Calculating class weights...")
        class_counts = np.array(class_counts_3)
        
        # Inverse frequency
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(weights)
        
        # Apply power scaling
        power = CONFIG['class_weight_power']
        weights = weights ** power
        weights = weights / weights.sum() * len(weights)
        
        # Extra boost for negative class (teacher priority!)
        negative_boost = CONFIG.get('negative_class_boost', 1.0)
        weights[0] = weights[0] * negative_boost
        
        # Re-normalize
        weights = weights / weights.sum() * len(weights)
        
        class_weights = torch.tensor(weights, dtype=torch.float32)
        
        print("Class weights (higher = more important):")
        for i, (name, w) in enumerate(zip(CONFIG['class_names'], class_weights)):
            bar = "█" * int(w * 15)
            boost_note = " ← BOOSTED (teacher priority)" if i == 0 else ""
            print(f"  {name}: {w:.4f} {bar}{boost_note}")
        print()
    else:
        class_weights = None
    
    # ============================================================
    # TRAIN / VAL / TEST SPLIT
    # ============================================================
    
    print("SPLITTING DATA")
    print("-" * 70)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['reviews'].values, df['label'].values,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['seed'],
        stratify=df['label'].values
    )
    
    val_ratio = CONFIG['val_size'] / (CONFIG['train_size'] + CONFIG['val_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=CONFIG['seed'],
        stratify=y_temp
    )
    
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print()
    
    del df
    gc.collect()
    
    # ============================================================
    # TOKENIZATION
    # ============================================================
    
    print("TOKENIZATION")
    print("-" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], local_files_only=True)
    
    train_ids, train_masks = tokenize_batch(X_train, tokenizer, CONFIG['max_length'], "Train")
    val_ids, val_masks = tokenize_batch(X_val, tokenizer, CONFIG['max_length'], "Val")
    test_ids, test_masks = tokenize_batch(X_test, tokenizer, CONFIG['max_length'], "Test")
    
    train_labels = torch.tensor(y_train, dtype=torch.long)
    val_labels = torch.tensor(y_val, dtype=torch.long)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    
    del X_train, X_val, X_test, y_train, y_val, y_test, X_temp, y_temp
    gc.collect()
    
    print()
    
    # ============================================================
    # DATALOADERS
    # ============================================================
    
    train_dataset = TensorDataset(train_ids, train_masks, train_labels)
    val_dataset = TensorDataset(val_ids, val_masks, val_labels)
    test_dataset = TensorDataset(test_ids, test_masks, test_labels)
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
        persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'],
        persistent_workers=True
    )
    
    print(f"Train batches: {len(train_loader):,}")
    print()
    
    # ============================================================
    # MODEL (3 classes!)
    # ============================================================
    
    print("LOADING MODEL")
    print("-" * 70)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=CONFIG['num_classes'],  # 3 classes!
        local_files_only=True
    )
    model = model.to(device)
    print(f"Model loaded with {CONFIG['num_classes']} output classes")
    print()
    
    # ============================================================
    # LOSS FUNCTION
    # ============================================================
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    criterion = FocalLoss(
        num_classes=CONFIG['num_classes'],
        gamma=CONFIG['focal_gamma'],
        alpha=class_weights,
        label_smoothing=CONFIG['label_smoothing']
    )
    
    print(f"Loss: Focal (γ={CONFIG['focal_gamma']}) + Label Smoothing ({CONFIG['label_smoothing']})")
    print()
    
    # ============================================================
    # OPTIMIZER & SCHEDULER
    # ============================================================
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    accum_steps = CONFIG['gradient_accumulation_steps']
    steps_per_epoch = len(train_loader) // accum_steps
    total_steps = steps_per_epoch * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    scheduler = get_scheduler(optimizer, CONFIG['scheduler_type'], total_steps, warmup_steps, CONFIG)
    scaler = torch.amp.GradScaler('cuda') if CONFIG['use_amp'] else None
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()
    
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    total_train_time = 0
    
    try:
        for epoch in range(CONFIG['epochs']):
            epoch_start = time.time()
            
            # === TRAIN ===
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", ncols=120)
            
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(pbar):
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                if CONFIG['use_amp']:
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits, labels) / accum_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels) / accum_steps
                    loss.backward()
                
                if (batch_idx + 1) % accum_steps == 0:
                    if CONFIG['use_amp']:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accum_steps
                _, pred = outputs.logits.max(1)
                train_total += labels.size(0)
                train_correct += pred.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{train_loss/(batch_idx+1):.4f}',
                    'acc': f'{100*train_correct/train_total:.1f}%'
                })
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # === VALIDATION ===
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            
            with torch.no_grad():
                for input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", ncols=120):
                    input_ids = input_ids.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    if CONFIG['use_amp']:
                        with torch.amp.autocast('cuda'):
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            loss = criterion(outputs.logits, labels)
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits, labels)
                    
                    val_loss += loss.item()
                    _, pred = outputs.logits.max(1)
                    val_total += labels.size(0)
                    val_correct += pred.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(scheduler.get_last_lr()[0])
            
            # === EPOCH SUMMARY ===
            print()
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}% | Time={epoch_time:.0f}s")
            
            # Checkpointing
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if CONFIG['checkpoint_every_epoch']:
                ckpt_path = os.path.join(CONFIG['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pt')
                save_checkpoint(model, tokenizer, optimizer, scheduler, scaler,
                                epoch+1, val_acc, val_loss, history, CONFIG, ckpt_path, is_best)
                cleanup_old_checkpoints(CONFIG['checkpoint_dir'], CONFIG['save_total_limit'])
            
            if is_best:
                print(f"  🏆 New best model saved!")
            
            if CONFIG['early_stopping'] and patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\n  🛑 Early stopping after {epoch+1} epochs")
                break
            
            print()
    
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        emergency_dir = CONFIG['output_dir'] + '_emergency'
        os.makedirs(emergency_dir, exist_ok=True)
        model.save_pretrained(emergency_dir)
        tokenizer.save_pretrained(emergency_dir)
        raise
    
    print("=" * 70)
    print(f"TRAINING COMPLETE - {total_train_time/60:.1f} minutes")
    print("=" * 70)
    print()
    
    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================
    
    print("FINAL TEST EVALUATION")
    print("-" * 70)
    
    # Load best model (now works without fix!)
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['output_dir'], local_files_only=True
    )
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc="Testing"):
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
    
    # Classification Report
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
    
    # Negative class recall (MOST IMPORTANT for teachers)
    negative_recall = report['Negative']['recall'] * 100
    negative_precision = report['Negative']['precision'] * 100
    
    print(f"  🔴 NEGATIVE FEEDBACK DETECTION (Struggling Students):")
    print(f"     Recall:    {negative_recall:.1f}% ← {negative_recall:.0f}% of struggling students caught")
    print(f"     Precision: {negative_precision:.1f}% ← {negative_precision:.0f}% of flags are real issues")
    print()
    
    # False negative analysis (missed struggling students)
    false_negatives = ((all_labels == 0) & (all_preds != 0)).sum()
    total_negatives = (all_labels == 0).sum()
    missed_pct = 100 * false_negatives / total_negatives
    
    print(f"  ⚠️  MISSED STRUGGLING STUDENTS:")
    print(f"     {false_negatives:,} of {total_negatives:,} negative cases missed ({missed_pct:.1f}%)")
    print()
    
    # Confidence analysis
    pred_confidence = all_probs.max(axis=1)
    low_confidence = (pred_confidence < 0.7).sum()
    low_conf_pct = 100 * low_confidence / len(pred_confidence)
    
    print(f"  🤔 UNCERTAIN PREDICTIONS (confidence < 70%):")
    print(f"     {low_confidence:,} of {len(pred_confidence):,} predictions ({low_conf_pct:.1f}%)")
    print(f"     → These should be flagged for manual review")
    print()
    
    # ============================================================
    # PLOTS
    # ============================================================
    
    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CONFIG['class_names'],
                yticklabels=CONFIG['class_names'], ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=CONFIG['class_names'],
                yticklabels=CONFIG['class_names'], ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix (Recall)')
    
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_3class.png', dpi=150)
    print("✓ Saved: plots/confusion_matrix_3class.png")
    
    # Per-class metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    width = 0.25
    
    recalls = [report[c]['recall']*100 for c in CONFIG['class_names']]
    precisions = [report[c]['precision']*100 for c in CONFIG['class_names']]
    f1s = [report[c]['f1-score']*100 for c in CONFIG['class_names']]
    
    bars1 = ax.bar(x - width, recalls, width, label='Recall', color='#e74c3c')
    bars2 = ax.bar(x, precisions, width, label='Precision', color='#3498db')
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#2ecc71')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Per-Class Metrics (3-Class Model)')
    ax.set_xticks(x)
    ax.set_xticklabels(['🔴 Negative\n(Needs Attention)', '🟡 Neutral\n(Mixed)', '🟢 Positive\n(Satisfied)'])
    ax.legend()
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/per_class_metrics_3class.png', dpi=150)
    print("✓ Saved: plots/per_class_metrics_3class.png")
    
    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs_range, history['train_loss'], 'b-o', label='Train')
    axes[0].plot(epochs_range, history['val_loss'], 'r-o', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs_range, history['train_acc'], 'b-o', label='Train')
    axes[1].plot(epochs_range, history['val_acc'], 'r-o', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/training_history_3class.png', dpi=150)
    print("✓ Saved: plots/training_history_3class.png")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    
    results = {
        'test_accuracy': test_acc,
        'negative_recall': negative_recall,
        'negative_precision': negative_precision,
        'missed_struggling_students': int(false_negatives),
        'total_negative_cases': int(total_negatives),
        'low_confidence_predictions': int(low_confidence),
        'config': CONFIG,
        'classification_report': report,
        'training_time_minutes': total_train_time / 60,
    }
    
    torch.save(results, os.path.join(CONFIG['output_dir'], 'results.pt'))
    
    with open(os.path.join(CONFIG['output_dir'], 'results.json'), 'w') as f:
        save_results = {k: v for k, v in results.items() if k not in ['config', 'classification_report']}
        save_results['per_class_recall'] = {c: report[c]['recall'] for c in CONFIG['class_names']}
        json.dump(save_results, f, indent=2)
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print()
    print("=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print(f"  Model saved to: {CONFIG['output_dir']}/")
    print()
    print("  RESULTS:")
    print(f"    Test Accuracy:     {test_acc:.1f}%")
    print(f"    Negative Recall:   {negative_recall:.1f}% ← Catches {negative_recall:.0f}% of struggling students")
    print(f"    Negative Precision: {negative_precision:.1f}%")
    print()
    print("  PER-CLASS RECALL:")
    for name in CONFIG['class_names']:
        recall = report[name]['recall'] * 100
        emoji = '🔴' if name == 'Negative' else ('🟡' if name == 'Neutral' else '🟢')
        print(f"    {emoji} {name}: {recall:.1f}%")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()