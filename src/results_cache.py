"""
Results caching system to avoid re-running experiments.
"""

import os
import json
import pickle
import hashlib
from typing import Any, Dict, Optional
from datetime import datetime


class ResultsCache:
    """Simple results caching system."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize cache directory."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, experiment_params: Dict[str, Any]) -> str:
        """Generate cache key from experiment parameters."""
        # Create a deterministic hash of the parameters
        param_str = json.dumps(experiment_params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, file_type: str = "json") -> str:
        """Get full path for cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.{file_type}")
    
    def save_results(self, experiment_params: Dict[str, Any], 
                    results: Dict[str, Any]) -> str:
        """Save experiment results to cache."""
        cache_key = self._get_cache_key(experiment_params)
        
        # Add metadata
        cached_data = {
            'experiment_params': experiment_params,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'cache_key': cache_key
        }
        
        # Save as JSON
        json_path = self._get_cache_path(cache_key, "json")
        with open(json_path, 'w') as f:
            json.dump(cached_data, f, indent=2, default=str)
        
        # Also save as pickle for complex objects
        pickle_path = self._get_cache_path(cache_key, "pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(cached_data, f)
        
        print(f"Results cached with key: {cache_key}")
        return cache_key
    
    def load_results(self, experiment_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load experiment results from cache."""
        cache_key = self._get_cache_key(experiment_params)
        json_path = self._get_cache_path(cache_key, "json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    cached_data = json.load(f)
                print(f"Loaded cached results: {cache_key}")
                return cached_data['results']
            except Exception as e:
                print(f"Error loading cache {cache_key}: {e}")
                return None
        
        return None
    
    def cache_exists(self, experiment_params: Dict[str, Any]) -> bool:
        """Check if cached results exist."""
        cache_key = self._get_cache_key(experiment_params)
        json_path = self._get_cache_path(cache_key, "json")
        return os.path.exists(json_path)
    
    def list_cached_experiments(self) -> list:
        """List all cached experiments."""
        cached_files = []
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    cached_files.append({
                        'cache_key': data.get('cache_key', filename[:-5]),
                        'timestamp': data.get('timestamp', 'unknown'),
                        'experiment_params': data.get('experiment_params', {})
                    })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        return sorted(cached_files, key=lambda x: x['timestamp'], reverse=True)
    
    def clear_cache(self):
        """Clear all cached results."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        print("Cache cleared")


def run_cached_experiment(experiment_func, experiment_params: Dict[str, Any], 
                         cache: Optional[ResultsCache] = None, 
                         force_rerun: bool = False) -> Dict[str, Any]:
    """
    Run experiment with caching.
    
    Args:
        experiment_func: Function that runs the experiment
        experiment_params: Parameters for the experiment
        cache: Cache instance (creates default if None)
        force_rerun: If True, ignore cache and rerun
        
    Returns:
        Experiment results
    """
    if cache is None:
        cache = ResultsCache()
    
    # Check cache first
    if not force_rerun and cache.cache_exists(experiment_params):
        print("Loading results from cache...")
        results = cache.load_results(experiment_params)
        if results is not None:
            return results
    
    # Run experiment
    print("Running experiment...")
    results = experiment_func(**experiment_params)
    
    # Cache results
    cache.save_results(experiment_params, results)
    
    return results
