"""
Train a logistic regression classifier on perplexity features.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib


def train_classifier():
    """Train classifier on perplexity delta features"""
    print("="*60)
    print("TRAINING CONSISTENCY CLASSIFIER")
    print("="*60)
    
    ROOT = Path(__file__).resolve().parents[2]
    scores_path = ROOT / "outputs" / "train_scores.csv"
    
    # Load scores
    df = pd.read_csv(scores_path)
    print(f"Loaded {len(df)} training examples")
    
    # Prepare features
    feature_cols = ['delta', 'ppl_ratio', 'baseline_loss', 'primed_loss']
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"\nFeatures: {feature_cols}")
    print(f"X shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Train logistic regression
    print("\nTraining logistic regression...")
    clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    clf.fit(X, y)
    
    # Predictions
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)
    
    # Evaluation
    print("\n" + "="*60)
    print("TRAINING SET PERFORMANCE")
    print("="*60)
    
    print(f"\nAccuracy: {accuracy_score(y, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Contradict', 'Consistent']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    print("                 Predicted")
    print("               Contra  Consist")
    print(f"Actual Contra    {cm[0,0]:3d}     {cm[0,1]:3d}")
    print(f"       Consist   {cm[1,0]:3d}     {cm[1,1]:3d}")
    
    # Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-fold)")
    print("="*60)
    
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance (coefficients)
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    for feat, coef in zip(feature_cols, clf.coef_[0]):
        print(f"  {feat:20s}: {coef:+.4f}")
    
    print(f"\n  Interpretation:")
    print(f"  - Positive coef → higher value predicts CONSISTENT (1)")
    print(f"  - Negative coef → higher value predicts CONTRADICT (0)")
    
    # Save classifier
    model_path = ROOT / "models" / "consistency_classifier.pkl"
    joblib.dump(clf, model_path)
    print(f"\n✅ Classifier saved to {model_path}")
    
    # Save predictions for analysis
    df['predicted'] = y_pred
    df['confidence'] = y_proba.max(axis=1)
    
    analysis_path = ROOT / "outputs" / "train_predictions.csv"
    df.to_csv(analysis_path, index=False)
    print(f"✅ Predictions saved to {analysis_path}")
    
    return clf, df


def analyze_errors(df):
    """Analyze misclassified examples"""
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    errors = df[df['label'] != df['predicted']]
    print(f"\nMisclassified: {len(errors)} / {len(df)} ({len(errors)/len(df)*100:.1f}%)")
    
    if len(errors) > 0:
        print("\nSample errors:")
        for idx, row in errors.head(5).iterrows():
            print(f"\nID {row['id']}: {row['char']} ({row['book_name']})")
            print(f"  True: {'Consistent' if row['label']==1 else 'Contradict'}")
            print(f"  Pred: {'Consistent' if row['predicted']==1 else 'Contradict'}")
            print(f"  Delta: {row['delta']:.4f}, PPL ratio: {row['ppl_ratio']:.3f}")


if __name__ == "__main__":
    clf, df = train_classifier()
    analyze_errors(df)

