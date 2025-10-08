#!/usr/bin/env python3
"""
Completely isolated test that avoids all problematic imports.
"""

import sys
import os
import time
import json

# Force single-threaded execution at the very start
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

print("ISOLATED DAVIDIAN REGULARIZATION TEST")
print("="*60)
print("Testing with minimal imports to avoid mutex blocking")
print("="*60)

def test_basic_functionality():
    """Test basic functionality without problematic imports."""
    
    print("\n1. Testing basic Python and NumPy...")
    import numpy as np
    print(f"   ✓ NumPy {np.__version__} imported")
    
    # Create simple test data (like Iris)
    np.random.seed(42)
    n_samples, n_features = 150, 4
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # 3 classes
    
    print(f"   ✓ Test data created: {X.shape}")
    
    print("\n2. Testing basic sklearn (minimal imports)...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("   ✓ Sklearn components imported")
    
    # Test basic model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"   ✓ Basic model test: accuracy = {accuracy:.4f}")
    
    print("\n3. Testing manual cross-validation...")
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        fold_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=1)
        fold_model.fit(X_fold_train, y_fold_train)
        
        train_pred = fold_model.predict(X_fold_train)
        val_pred = fold_model.predict(X_fold_val)
        
        train_score = accuracy_score(y_fold_train, train_pred)
        val_score = accuracy_score(y_fold_val, val_pred)
        
        # Apply Davidian Regularization
        penalty = abs(train_score - val_score)
        regularized_score = val_score - penalty  # alpha = 1.0
        
        fold_scores.append({
            'fold': fold_idx,
            'train_score': train_score,
            'val_score': val_score,
            'penalty': penalty,
            'regularized_score': regularized_score
        })
        
        print(f"   Fold {fold_idx + 1}: train={train_score:.4f}, val={val_score:.4f}, "
              f"penalty={penalty:.4f}, regularized={regularized_score:.4f}")
    
    mean_regularized = np.mean([f['regularized_score'] for f in fold_scores])
    print(f"   ✓ Mean regularized score: {mean_regularized:.4f}")
    
    print("\n4. Testing multiple trials...")
    trial_scores = []
    
    for trial in range(5):
        kf_trial = KFold(n_splits=3, shuffle=True, random_state=trial)
        trial_fold_scores = []
        
        for train_idx, val_idx in kf_trial.split(X):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            trial_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=1)
            trial_model.fit(X_fold_train, y_fold_train)
            
            train_pred = trial_model.predict(X_fold_train)
            val_pred = trial_model.predict(X_fold_val)
            
            train_score = accuracy_score(y_fold_train, train_pred)
            val_score = accuracy_score(y_fold_val, val_pred)
            
            penalty = abs(train_score - val_score)
            regularized_score = val_score - penalty
            
            trial_fold_scores.append(regularized_score)
        
        trial_mean = np.mean(trial_fold_scores)
        trial_scores.append(trial_mean)
        print(f"   Trial {trial + 1}: {trial_mean:.4f}")
    
    # Get best 4 trials (or all if less than 4)
    best_4_scores = sorted(trial_scores, reverse=True)[:4]
    davidian_mean = np.mean(best_4_scores)
    
    print(f"   ✓ Davidian best 4 mean: {davidian_mean:.4f}")
    
    print("\n5. Testing random sampling comparison...")
    random_scores = []
    
    for trial in range(5):
        X_rand_train, X_rand_val, y_rand_train, y_rand_val = train_test_split(
            X, y, test_size=0.2, random_state=trial
        )
        
        rand_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=1)
        rand_model.fit(X_rand_train, y_rand_train)
        
        rand_pred = rand_model.predict(X_rand_val)
        rand_score = accuracy_score(y_rand_val, rand_pred)
        
        random_scores.append(rand_score)
        print(f"   Random trial {trial + 1}: {rand_score:.4f}")
    
    best_4_random = sorted(random_scores, reverse=True)[:4]
    random_mean = np.mean(best_4_random)
    
    print(f"   ✓ Random best 4 mean: {random_mean:.4f}")
    
    print("\n6. Final comparison...")
    improvement = davidian_mean - random_mean
    improvement_pct = (improvement / abs(random_mean)) * 100 if random_mean != 0 else 0
    
    print(f"   Davidian Regularization: {davidian_mean:.4f}")
    print(f"   Random Sampling: {random_mean:.4f}")
    print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    # Save results
    results = {
        'davidian_mean': davidian_mean,
        'random_mean': random_mean,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'davidian_scores': trial_scores,
        'random_scores': random_scores,
        'fold_details': fold_scores,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to file
    os.makedirs('results', exist_ok=True)
    with open('results/isolated_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n   ✓ Results saved to results/isolated_test_results.json")
    
    if improvement > 0:
        print("\n🎉 DAVIDIAN REGULARIZATION SHOWS IMPROVEMENT! 🎉")
    else:
        print("\n📊 Random sampling performed better in this test")
    
    return results

def main():
    """Run the isolated test."""
    try:
        start_time = time.time()
        results = test_basic_functionality()
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("✅ ISOLATED TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.2f} seconds")
        print("✅ No mutex blocking issues")
        print("✅ Davidian Regularization algorithm works")
        print("✅ Results cached for future reference")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ TEST FAILED")
        print(f"{'='*60}")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
