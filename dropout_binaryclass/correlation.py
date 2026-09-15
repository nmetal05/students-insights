#!/usr/bin/env python
# coding: utf-8
"""
Feature Correlation Analysis
Helps identify redundant features and features most correlated with Target.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = pd.read_csv('data.csv', sep=';')
df = df[df['Target'] != 'Enrolled']
df['Target'] = df['Target'].map({'Dropout': 0, 'Graduate': 1})

print(f"Dataset shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")

# =============================================================================
# 2. CORRELATION WITH TARGET
# =============================================================================

print("\n" + "="*70)
print("CORRELATION WITH TARGET (Dropout=0, Graduate=1)")
print("="*70)

# Calculate correlation with target
target_corr = df.corr()['Target'].drop('Target').sort_values(key=abs, ascending=False)

print("\nAll features ranked by absolute correlation with Target:\n")
for i, (feature, corr) in enumerate(target_corr.items(), 1):
    strength = "STRONG" if abs(corr) > 0.3 else "MODERATE" if abs(corr) > 0.15 else "WEAK"
    print(f"{i:2d}. {feature:50s} {corr:+.4f}  [{strength}]")

# Plot correlation with target
plt.figure(figsize=(12, 10))
colors = ['green' if c > 0 else 'red' for c in target_corr.values]
target_corr.plot(kind='barh', color=colors)
plt.title('Feature Correlation with Target (Graduate=1)')
plt.xlabel('Correlation Coefficient')
plt.axvline(x=0, color='black', linewidth=0.5)
plt.axvline(x=0.3, color='blue', linestyle='--', alpha=0.5, label='Strong threshold')
plt.axvline(x=-0.3, color='blue', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('correlation_with_target.png', dpi=150)
plt.show()

# =============================================================================
# 3. FEATURE-TO-FEATURE CORRELATION (Find Redundant Features)
# =============================================================================

print("\n" + "="*70)
print("HIGHLY CORRELATED FEATURE PAIRS (Potential Redundancy)")
print("="*70)

# Calculate correlation matrix
corr_matrix = df.drop('Target', axis=1).corr()

# Find highly correlated pairs
high_corr_pairs = []
threshold = 0.7

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) >= threshold:
            high_corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_value
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)

print(f"\nFeature pairs with correlation >= {threshold}:\n")
if len(high_corr_df) > 0:
    for _, row in high_corr_df.iterrows():
        print(f"  {row['Correlation']:+.4f}  |  {row['Feature 1']}")
        print(f"           |  {row['Feature 2']}")
        print()
else:
    print("  No highly correlated pairs found.")

# =============================================================================
# 4. CORRELATION HEATMAP
# =============================================================================

plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5,
            annot_kws={'size': 6})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()

# =============================================================================
# 5. RECOMMENDATIONS FOR FEATURE SELECTION
# =============================================================================

print("\n" + "="*70)
print("FEATURE SELECTION RECOMMENDATIONS")
print("="*70)

# Weak correlation with target (candidates for removal)
weak_threshold = 0.05
weak_features = target_corr[abs(target_corr) < weak_threshold]

print(f"\n1. WEAK CORRELATION WITH TARGET (|corr| < {weak_threshold}):")
print("   Consider removing these - they may not help prediction:\n")
for feature, corr in weak_features.items():
    print(f"   - {feature}: {corr:+.4f}")

# Features to keep (strong correlation)
strong_threshold = 0.2
strong_features = target_corr[abs(target_corr) >= strong_threshold]

print(f"\n2. STRONG CORRELATION WITH TARGET (|corr| >= {strong_threshold}):")
print("   Keep these - they are predictive:\n")
for feature, corr in strong_features.items():
    print(f"   + {feature}: {corr:+.4f}")

# Redundant features (high correlation with each other)
print(f"\n3. REDUNDANT FEATURES (correlated with each other >= {threshold}):")
print("   Consider keeping only one from each pair:\n")
for _, row in high_corr_df.iterrows():
    # Suggest keeping the one more correlated with target
    corr1 = abs(target_corr.get(row['Feature 1'], 0))
    corr2 = abs(target_corr.get(row['Feature 2'], 0))
    keep = row['Feature 1'] if corr1 >= corr2 else row['Feature 2']
    drop = row['Feature 2'] if corr1 >= corr2 else row['Feature 1']
    print(f"   KEEP: {keep} (target corr: {target_corr.get(keep, 0):+.4f})")
    print(f"   DROP: {drop} (target corr: {target_corr.get(drop, 0):+.4f})")
    print()

# =============================================================================
# 6. SUGGESTED FEATURES TO DROP
# =============================================================================

print("\n" + "="*70)
print("SUGGESTED FEATURES TO DROP")
print("="*70)

features_to_drop = set()

# Add weak features
for f in weak_features.index:
    features_to_drop.add(f)

# Add redundant features (the one less correlated with target)
for _, row in high_corr_df.iterrows():
    corr1 = abs(target_corr.get(row['Feature 1'], 0))
    corr2 = abs(target_corr.get(row['Feature 2'], 0))
    drop = row['Feature 2'] if corr1 >= corr2 else row['Feature 1']
    features_to_drop.add(drop)

print(f"\nBased on analysis, consider dropping these {len(features_to_drop)} features:\n")
for f in features_to_drop:
    reason = []
    if f in weak_features.index:
        reason.append(f"weak target corr ({target_corr[f]:+.4f})")
    if f in [row['Feature 1'] for _, row in high_corr_df.iterrows()] or \
       f in [row['Feature 2'] for _, row in high_corr_df.iterrows()]:
        reason.append("redundant with another feature")
    print(f"  - {f}")
    print(f"    Reason: {', '.join(reason)}")

# Features to keep
features_to_keep = [f for f in target_corr.index if f not in features_to_drop]

print(f"\nKeep these {len(features_to_keep)} features:\n")
for f in features_to_keep:
    print(f"  + {f} (target corr: {target_corr[f]:+.4f})")

# =============================================================================
# 7. GENERATE CODE SNIPPET
# =============================================================================

print("\n" + "="*70)
print("CODE SNIPPET FOR YOUR TRAINING SCRIPT")
print("="*70)

print("\n# Copy this to your training script:")
print(f"columns_to_drop = {list(features_to_drop)}")

# =============================================================================
# 8. SAVE ANALYSIS RESULTS
# =============================================================================

# Save correlation with target
target_corr.to_csv('target_correlations.csv', header=['correlation'])

# Save high correlation pairs
if len(high_corr_df) > 0:
    high_corr_df.to_csv('redundant_feature_pairs.csv', index=False)

# Save recommendations
with open('feature_selection_recommendations.txt', 'w') as f:
    f.write("FEATURE SELECTION RECOMMENDATIONS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Features to DROP ({len(features_to_drop)}):\n")
    for feat in features_to_drop:
        f.write(f"  - {feat}\n")
    f.write(f"\nFeatures to KEEP ({len(features_to_keep)}):\n")
    for feat in features_to_keep:
        f.write(f"  + {feat}\n")

print("\nFiles saved:")
print("  1. correlation_with_target.png")
print("  2. correlation_matrix.png")
print("  3. target_correlations.csv")
print("  4. redundant_feature_pairs.csv")
print("  5. feature_selection_recommendations.txt")