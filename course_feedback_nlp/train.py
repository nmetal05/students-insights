"""
Course Review Sentiment Model - Training Script
VRAM Optimized for AMD 7900 XTX (24GB)

PATCHES APPLIED:
- Class weights to handle imbalanced data (78.8% are 5-star reviews)
- Optimized batch_size=128 for better accuracy
- max_length=96 for faster training
- AMD crash protection and emergency checkpointing
- Periodic checkpoint saving (every epoch)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# AMD CRASH PROTECTION - Suppress problematic logging
# ============================================================
os.environ['AMD_LOG_LEVEL'] = '0'
os.environ['ROCM_LOG_LEVEL'] = '0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'data_path': 'Coursera_reviews.csv',
    'model_name': './distilbert-base-uncased',
    'output_dir': 'sentiment_model',
    'checkpoint_dir': 'checkpoints',      # NEW: For periodic saves
    'max_length': 96,                     # CHANGED: 128 → 96 (faster, minimal accuracy loss)
    'batch_size': 128,                    # CHANGED: 512 → 128 (better accuracy per Run 3)
    'epochs': 5,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'train_size': 0.8,
    'val_size': 0.1,
    'test_size': 0.1,
    'seed': 42,
    'num_workers': 4,
    'pin_memory': True,
    'use_amp': True,                      # Mixed precision for speed
    'use_class_weights': True,            # NEW: Address class imbalance
    'checkpoint_every_epoch': True,       # NEW: Save checkpoint every epoch
}

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    # ============================================================
    # SETUP
    # ============================================================
    
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    set_seed(CONFIG['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Memory: {total_mem:.2f} GB")
    print("=" * 70)
    print()
    
    # Create directories early
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # ============================================================
    # VERIFY LOCAL MODEL EXISTS
    # ============================================================
    
    print("=" * 70)
    print("VERIFYING LOCAL MODEL")
    print("=" * 70)
    
    model_path = CONFIG['model_name']
    if os.path.exists(model_path):
        print(f"  ✓ Model directory found: {model_path}")
    else:
        print(f"  ✗ Model directory NOT found: {model_path}")
        return
    
    print("=" * 70)
    print()
    
    # ============================================================
    # DATA LOADING
    # ============================================================
    
    print("=" * 70)
    print("DATA LOADING")
    print("=" * 70)
    
    print("Loading data...")
    df = pd.read_csv(CONFIG['data_path'])
    print(f"  Raw data shape: {df.shape}")
    
    # Clean data
    df = df.dropna(subset=['reviews', 'rating'])
    df = df[df['reviews'].str.strip() != '']
    df['rating'] = df['rating'].astype(int)
    df = df[df['rating'].between(1, 5)]
    df['label'] = df['rating'] - 1
    
    print(f"  Cleaned data shape: {df.shape}")
    print(f"\n  Rating distribution:")
    for rating, count in df['rating'].value_counts().sort_index().items():
        pct = 100 * count / len(df)
        bar = "█" * int(pct / 2)
        print(f"    {rating} Star: {count:>8,} ({pct:>5.1f}%) {bar}")
    
    # ============================================================
    # CALCULATE CLASS WEIGHTS (Before deleting df!)
    # ============================================================
    
    if CONFIG['use_class_weights']:
        print(f"\n  Calculating class weights...")
        class_counts = df['label'].value_counts().sort_index().values
        # Inverse frequency weighting
        class_weights = 1.0 / class_counts
        # Normalize so weights sum to num_classes
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        print(f"  Class weights (to balance {class_counts[-1]/class_counts[0]:.1f}x imbalance):")
        for i, (w, c) in enumerate(zip(class_weights, class_counts)):
            print(f"    {i+1} Star: weight={w:.4f} (count={c:,})")
    else:
        class_weights = None
    
    print("=" * 70)
    print()
    
    # ============================================================
    # TRAIN / VALIDATION / TEST SPLIT
    # ============================================================
    
    print("=" * 70)
    print("DATA SPLITTING")
    print("=" * 70)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['reviews'].values,
        df['label'].values,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['seed'],
        stratify=df['label'].values
    )
    
    relative_val_size = CONFIG['val_size'] / (CONFIG['train_size'] + CONFIG['val_size'])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=CONFIG['seed'],
        stratify=y_temp
    )
    
    print(f"  Training samples:   {len(X_train):>10,} ({100*len(X_train)/len(df):.1f}%)")
    print(f"  Validation samples: {len(X_val):>10,} ({100*len(X_val)/len(df):.1f}%)")
    print(f"  Test samples:       {len(X_test):>10,} ({100*len(X_test)/len(df):.1f}%)")
    print("=" * 70)
    print()
    
    # Now we can delete df
    del df
    gc.collect()
    
    # ============================================================
    # TOKENIZER
    # ============================================================
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_name'],
        local_files_only=True
    )
    print(f"  ✓ Tokenizer loaded")
    print()
    
    # ============================================================
    # PRE-TOKENIZE ALL DATA (Key optimization!)
    # ============================================================
    
    print("=" * 70)
    print("PRE-TOKENIZING ALL DATA")
    print("=" * 70)
    print("  This runs once and stores tensors for fast loading...")
    print()
    
    def tokenize_batch(texts, desc="Tokenizing"):
        """Tokenize all texts at once using batch processing"""
        all_input_ids = []
        all_attention_masks = []
        
        batch_size = 10000  # Process 10k at a time to avoid memory issues
        
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[i:i+batch_size].tolist()
            
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=CONFIG['max_length'],
                return_tensors='pt'
            )
            
            all_input_ids.append(encodings['input_ids'])
            all_attention_masks.append(encodings['attention_mask'])
        
        return (
            torch.cat(all_input_ids, dim=0),
            torch.cat(all_attention_masks, dim=0)
        )
    
    # Tokenize train
    print("  Tokenizing training data...")
    train_input_ids, train_attention_masks = tokenize_batch(X_train, "  Train")
    train_labels = torch.tensor(y_train, dtype=torch.long)
    
    # Tokenize validation
    print("  Tokenizing validation data...")
    val_input_ids, val_attention_masks = tokenize_batch(X_val, "  Val")
    val_labels = torch.tensor(y_val, dtype=torch.long)
    
    # Tokenize test
    print("  Tokenizing test data...")
    test_input_ids, test_attention_masks = tokenize_batch(X_test, "  Test")
    test_labels = torch.tensor(y_test, dtype=torch.long)
    
    # Free memory
    del X_train, X_val, X_test, y_train, y_val, y_test, X_temp, y_temp
    gc.collect()
    
    print()
    print(f"  ✓ Train tensors: {train_input_ids.shape}")
    print(f"  ✓ Val tensors:   {val_input_ids.shape}")
    print(f"  ✓ Test tensors:  {test_input_ids.shape}")
    print("=" * 70)
    print()
    
    # ============================================================
    # CREATE TENSOR DATASETS (Fast!)
    # ============================================================
    
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    
    # ============================================================
    # DATALOADERS
    # ============================================================
    
    print("Creating dataloaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True  # NEW: Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True
    )
    
    print(f"  ✓ Train batches:      {len(train_loader):,}")
    print(f"  ✓ Validation batches: {len(val_loader):,}")
    print(f"  ✓ Test batches:       {len(test_loader):,}")
    print()
    
    # ============================================================
    # MODEL
    # ============================================================
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=5,
        local_files_only=True
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model loaded")
    print(f"  ✓ Total parameters: {total_params:,}")
    print()
    
    # ============================================================
    # LOSS FUNCTION WITH CLASS WEIGHTS
    # ============================================================
    
    if CONFIG['use_class_weights'] and class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  ✓ Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"  ✓ Using standard CrossEntropyLoss")
    print()
    
    # ============================================================
    # OPTIMIZER & SCHEDULER
    # ============================================================
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed Precision Scaler (for speed)
    scaler = torch.amp.GradScaler('cuda') if CONFIG['use_amp'] else None
    
    print("Optimizer & Scheduler configured:")
    print(f"  ✓ Optimizer:        AdamW (lr={CONFIG['learning_rate']})")
    print(f"  ✓ Total steps:      {total_steps:,}")
    print(f"  ✓ Warmup steps:     {warmup_steps:,}")
    print(f"  ✓ Mixed Precision:  {CONFIG['use_amp']}")
    print()
    
    # ============================================================
    # HELPER FUNCTION: Save checkpoint
    # ============================================================
    
    def save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, epoch, 
                        val_acc, history, path, is_best=False):
        """Save a training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'val_accuracy': val_acc,
            'history': history,
            'config': CONFIG,
        }
        torch.save(checkpoint, path)
        
        if is_best:
            model.save_pretrained(CONFIG['output_dir'])
            tokenizer.save_pretrained(CONFIG['output_dir'])
    
    # ============================================================
    # TRAINING LOOP (with crash protection)
    # ============================================================
    
    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)
    print(f"  Epochs:        {CONFIG['epochs']}")
    print(f"  Batch size:    {CONFIG['batch_size']}")
    print(f"  Max length:    {CONFIG['max_length']}")
    print(f"  Device:        {device}")
    print(f"  AMP:           {CONFIG['use_amp']}")
    print(f"  Class weights: {CONFIG['use_class_weights']}")
    print("=" * 70)
    print()
    
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    total_train_time = 0
    
    # ============================================================
    # WRAP IN TRY/EXCEPT FOR CRASH PROTECTION
    # ============================================================
    
    try:
        for epoch in range(CONFIG['epochs']):
            epoch_start_time = time.time()
            
            # ==================== TRAINING ====================
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [TRAIN]",
                unit="batch",
                ncols=120
            )
            
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_pbar):
                # Move to GPU with non_blocking for speed
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if CONFIG['use_amp']:
                    with torch.amp.autocast('cuda'):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        # USE CUSTOM LOSS WITH CLASS WEIGHTS
                        logits = outputs.logits
                        loss = criterion(logits, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                running_loss = train_loss / (batch_idx + 1)
                running_acc = 100 * train_correct / train_total
                current_lr = scheduler.get_last_lr()[0]
                
                # Show GPU memory usage
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / 1e9
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                train_pbar.set_postfix({
                    'loss': f'{running_loss:.4f}',
                    'acc': f'{running_acc:.2f}%',
                    'lr': f'{current_lr:.2e}',
                    'VRAM': f'{mem_used:.1f}/{mem_total:.1f}GB'
                })
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # ==================== VALIDATION ====================
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            val_pbar = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [VAL]  ",
                unit="batch",
                ncols=120
            )
            
            with torch.no_grad():
                for batch_idx, (input_ids, attention_mask, labels) in enumerate(val_pbar):
                    input_ids = input_ids.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    if CONFIG['use_amp']:
                        with torch.amp.autocast('cuda'):
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )
                            logits = outputs.logits
                            loss = criterion(logits, labels)
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        logits = outputs.logits
                        loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    _, predicted = logits.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    running_loss = val_loss / (batch_idx + 1)
                    running_acc = 100 * val_correct / val_total
                    
                    val_pbar.set_postfix({
                        'loss': f'{running_loss:.4f}',
                        'acc': f'{running_acc:.2f}%'
                    })
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            total_train_time += epoch_time
            
            # ==================== EPOCH SUMMARY ====================
            print()
            print("─" * 70)
            print(f"EPOCH {epoch+1}/{CONFIG['epochs']} SUMMARY")
            print("─" * 70)
            print(f"  {'Metric':<20} {'Train':>15} {'Validation':>15}")
            print(f"  {'-'*20} {'-'*15} {'-'*15}")
            print(f"  {'Loss':<20} {train_loss:>15.4f} {val_loss:>15.4f}")
            print(f"  {'Accuracy':<20} {train_acc:>14.2f}% {val_acc:>14.2f}%")
            print(f"  {'-'*20} {'-'*15} {'-'*15}")
            print(f"  {'Time':<20} {epoch_time:>14.1f}s")
            print(f"  {'Samples/sec':<20} {len(train_dataset)/epoch_time:>14.1f}")
            
            # ==================== SAVE CHECKPOINT ====================
            is_best = val_acc > best_val_acc
            
            if is_best:
                best_val_acc = val_acc
            
            # Always save periodic checkpoint
            if CONFIG['checkpoint_every_epoch']:
                checkpoint_path = os.path.join(
                    CONFIG['checkpoint_dir'], 
                    f'checkpoint_epoch_{epoch+1}.pt'
                )
                save_checkpoint(
                    model, tokenizer, optimizer, scheduler, scaler,
                    epoch + 1, val_acc, history, checkpoint_path, is_best=is_best
                )
                print(f"\n  💾 Checkpoint saved: {checkpoint_path}")
            
            if is_best:
                # Also save as best model
                torch.save({
                    'epoch': epoch + 1,
                    'best_val_accuracy': best_val_acc,
                    'config': CONFIG,
                    'history': history
                }, os.path.join(CONFIG['output_dir'], 'training_info.pt'))
                
                print(f"  🏆 NEW BEST MODEL SAVED! Val Accuracy: {best_val_acc:.2f}%")
            else:
                print(f"\n  ℹ️  Best Val Accuracy so far: {best_val_acc:.2f}%")
            
            print("─" * 70)
            print()
    
    except Exception as e:
        # ============================================================
        # EMERGENCY SAVE ON CRASH
        # ============================================================
        print()
        print("!" * 70)
        print("⚠️  ERROR OCCURRED - SAVING EMERGENCY CHECKPOINT")
        print("!" * 70)
        print(f"  Error: {e}")
        
        emergency_dir = CONFIG['output_dir'] + '_emergency'
        os.makedirs(emergency_dir, exist_ok=True)
        
        try:
            model.save_pretrained(emergency_dir)
            tokenizer.save_pretrained(emergency_dir)
            
            torch.save({
                'epoch': epoch + 1 if 'epoch' in dir() else 0,
                'history': history,
                'config': CONFIG,
                'error': str(e)
            }, os.path.join(emergency_dir, 'emergency_checkpoint.pt'))
            
            print(f"  ✓ Emergency checkpoint saved to: {emergency_dir}")
        except Exception as save_error:
            print(f"  ✗ Failed to save emergency checkpoint: {save_error}")
        
        print("!" * 70)
        raise  # Re-raise the exception
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total training time: {total_train_time/60:.1f} minutes")
    print(f"  Best Val Accuracy:   {best_val_acc:.2f}%")
    print("=" * 70)
    print()
    
    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================
    
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print("Loading best model...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['output_dir'],
        local_files_only=True
    )
    model = model.to(device)
    model.eval()
    
    # Use standard loss for test evaluation (no class weights)
    test_criterion = nn.CrossEntropyLoss()
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    test_pbar = tqdm(test_loader, desc="Testing", unit="batch", ncols=120)
    
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(test_pbar):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if CONFIG['use_amp']:
                with torch.amp.autocast('cuda'):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    loss = test_criterion(logits, labels)
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                loss = test_criterion(logits, labels)
            
            test_loss += loss.item()
            _, predicted = logits.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            test_pbar.set_postfix({
                'loss': f'{test_loss/(batch_idx+1):.4f}',
                'acc': f'{100*test_correct/test_total:.2f}%'
            })
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    within_one = np.mean(np.abs(all_preds - all_labels) <= 1) * 100
    
    print()
    print("─" * 70)
    print("TEST RESULTS")
    print("─" * 70)
    print(f"  Test Loss:              {test_loss:.4f}")
    print(f"  Test Accuracy:          {test_acc:.2f}%")
    print(f"  Within ±1 Star:         {within_one:.2f}%")
    print("─" * 70)
    print()
    
    print("CLASSIFICATION REPORT")
    print("─" * 70)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=['1 Star', '2 Star', '3 Star', '4 Star', '5 Star'],
        digits=3,
        output_dict=True
    )
    print(classification_report(
        all_labels,
        all_preds,
        target_names=['1 Star', '2 Star', '3 Star', '4 Star', '5 Star'],
        digits=3
    ))
    
    # ============================================================
    # PLOTS
    # ============================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs_range, history['train_loss'], 'b-o', label='Train', linewidth=2)
    axes[0].plot(epochs_range, history['val_loss'], 'r-o', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss (with Class Weights)' if CONFIG['use_class_weights'] else 'Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs_range, history['train_acc'], 'b-o', label='Train', linewidth=2)
    axes[1].plot(epochs_range, history['val_acc'], 'r-o', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png', dpi=150)
    print("✓ Saved: plots/training_history.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cm = confusion_matrix(all_labels, all_preds)
    labels_names = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_names, yticklabels=labels_names, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=labels_names, yticklabels=labels_names, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    print("✓ Saved: plots/confusion_matrix.png")
    
    # ============================================================
    # PER-CLASS RECALL COMPARISON PLOT (NEW!)
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star']
    recalls = [report[c]['recall'] * 100 for c in classes]
    
    bars = ax.bar(classes, recalls, color=['#ff6b6b', '#ffa94d', '#ffd43b', '#69db7c', '#4dabf7'])
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.axhline(y=75, color='green', linestyle='--', alpha=0.5, label='75% threshold')
    
    for bar, recall in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{recall:.1f}%', ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('Recall (%)')
    ax.set_title('Per-Class Recall (Higher = Better at detecting this class)')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/per_class_recall.png', dpi=150)
    print("✓ Saved: plots/per_class_recall.png")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    
    results = {
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'test_within_one': within_one,
        'history': history,
        'config': CONFIG,
        'train_time_minutes': total_train_time / 60,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    torch.save(results, os.path.join(CONFIG['output_dir'], 'results.pt'))
    
    print()
    print("=" * 70)
    print("🎉 ALL DONE!")
    print("=" * 70)
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  Test Accuracy:     {test_acc:.2f}%")
    print(f"  Within ±1 Star:    {within_one:.2f}%")
    print(f"  Training Time:     {total_train_time/60:.1f} minutes")
    print()
    print("  Per-Class Recall:")
    for c in classes:
        recall = report[c]['recall'] * 100
        indicator = "✓" if recall >= 60 else "⚠️" if recall >= 40 else "✗"
        print(f"    {indicator} {c}: {recall:.1f}%")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    main()