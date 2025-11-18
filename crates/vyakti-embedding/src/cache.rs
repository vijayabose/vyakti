//! Embedding cache for recomputation.

use dashmap::DashMap;
use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use vyakti_common::Vector;

/// LRU cache for embeddings.
///
/// Thread-safe cache that stores embeddings with a maximum capacity,
/// evicting least recently used items when full.
pub struct EmbeddingCache<K: Hash + Eq + Clone> {
    /// Cached embeddings
    cache: DashMap<K, Vector>,
    /// LRU queue for tracking access order
    lru_queue: Arc<Mutex<VecDeque<K>>>,
    /// Maximum cache size
    capacity: usize,
    /// Cache hit count
    hits: Arc<Mutex<usize>>,
    /// Cache miss count
    misses: Arc<Mutex<usize>>,
}

impl<K: Hash + Eq + Clone> EmbeddingCache<K> {
    /// Create a new embedding cache with the given capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of embeddings to store
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: DashMap::new(),
            lru_queue: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
            hits: Arc::new(Mutex::new(0)),
            misses: Arc::new(Mutex::new(0)),
        }
    }

    /// Get an embedding from the cache.
    ///
    /// # Arguments
    ///
    /// * `key` - Cache key
    ///
    /// # Returns
    ///
    /// `Some(embedding)` if found, `None` otherwise
    pub fn get(&self, key: &K) -> Option<Vector> {
        if let Some(embedding) = self.cache.get(key) {
            // Update LRU queue
            let mut queue = self.lru_queue.lock().unwrap();
            queue.retain(|k| k != key);
            queue.push_back(key.clone());
            drop(queue);

            // Increment hit count
            *self.hits.lock().unwrap() += 1;

            Some(embedding.clone())
        } else {
            // Increment miss count
            *self.misses.lock().unwrap() += 1;
            None
        }
    }

    /// Insert an embedding into the cache.
    ///
    /// # Arguments
    ///
    /// * `key` - Cache key
    /// * `embedding` - Embedding vector
    pub fn insert(&self, key: K, embedding: Vector) {
        // If cache is at capacity, evict LRU item
        if self.cache.len() >= self.capacity {
            let mut queue = self.lru_queue.lock().unwrap();
            if let Some(lru_key) = queue.pop_front() {
                self.cache.remove(&lru_key);
            }
            drop(queue);
        }

        // Insert new item
        self.cache.insert(key.clone(), embedding);

        // Update LRU queue
        let mut queue = self.lru_queue.lock().unwrap();
        queue.retain(|k| k != &key);
        queue.push_back(key);
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) {
        self.cache.clear();
        self.lru_queue.lock().unwrap().clear();
        *self.hits.lock().unwrap() = 0;
        *self.misses.lock().unwrap() = 0;
    }

    /// Get the current cache size.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache hit count.
    pub fn hits(&self) -> usize {
        *self.hits.lock().unwrap()
    }

    /// Get cache miss count.
    pub fn misses(&self) -> usize {
        *self.misses.lock().unwrap()
    }

    /// Get cache hit rate (hits / (hits + misses)).
    ///
    /// Returns 0.0 if no requests have been made.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits();
        let misses = self.misses();
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_new() {
        let cache: EmbeddingCache<String> = EmbeddingCache::new(10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_cache_insert_and_get() {
        let cache = EmbeddingCache::new(10);
        let key = "test".to_string();
        let embedding = vec![1.0, 2.0, 3.0];

        cache.insert(key.clone(), embedding.clone());

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved, embedding);

        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_cache_miss() {
        let cache: EmbeddingCache<String> = EmbeddingCache::new(10);
        let result = cache.get(&"nonexistent".to_string());

        assert!(result.is_none());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let cache = EmbeddingCache::new(3);

        // Insert 3 items
        cache.insert("a".to_string(), vec![1.0]);
        cache.insert("b".to_string(), vec![2.0]);
        cache.insert("c".to_string(), vec![3.0]);

        assert_eq!(cache.len(), 3);

        // Insert 4th item, should evict "a"
        cache.insert("d".to_string(), vec![4.0]);

        assert_eq!(cache.len(), 3);
        assert!(cache.get(&"a".to_string()).is_none());
        assert!(cache.get(&"b".to_string()).is_some());
        assert!(cache.get(&"c".to_string()).is_some());
        assert!(cache.get(&"d".to_string()).is_some());
    }

    #[test]
    fn test_cache_lru_update() {
        let cache = EmbeddingCache::new(3);

        cache.insert("a".to_string(), vec![1.0]);
        cache.insert("b".to_string(), vec![2.0]);
        cache.insert("c".to_string(), vec![3.0]);

        // Access "a" to make it most recently used
        let _ = cache.get(&"a".to_string());

        // Insert "d", should evict "b" (least recently used)
        cache.insert("d".to_string(), vec![4.0]);

        assert!(cache.get(&"a".to_string()).is_some());
        assert!(cache.get(&"b".to_string()).is_none());
        assert!(cache.get(&"c".to_string()).is_some());
        assert!(cache.get(&"d".to_string()).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let cache = EmbeddingCache::new(10);

        cache.insert("a".to_string(), vec![1.0]);
        cache.insert("b".to_string(), vec![2.0]);
        let _ = cache.get(&"a".to_string());

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.hits(), 1);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_cache_hit_rate() {
        let cache = EmbeddingCache::new(10);

        // No requests yet
        assert_eq!(cache.hit_rate(), 0.0);

        cache.insert("a".to_string(), vec![1.0]);
        cache.insert("b".to_string(), vec![2.0]);

        // 2 hits
        let _ = cache.get(&"a".to_string());
        let _ = cache.get(&"b".to_string());

        // 1 miss
        let _ = cache.get(&"c".to_string());

        // Hit rate should be 2/3
        assert!((cache.hit_rate() - (2.0 / 3.0)).abs() < 1e-6);

        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_cache_duplicate_insert() {
        let cache = EmbeddingCache::new(10);

        cache.insert("a".to_string(), vec![1.0]);
        cache.insert("a".to_string(), vec![2.0]); // Overwrite

        assert_eq!(cache.len(), 1);
        let embedding = cache.get(&"a".to_string()).unwrap();
        assert_eq!(embedding, vec![2.0]);
    }

    #[test]
    fn test_cache_capacity_one() {
        let cache = EmbeddingCache::new(1);

        cache.insert("a".to_string(), vec![1.0]);
        assert_eq!(cache.len(), 1);

        cache.insert("b".to_string(), vec![2.0]);
        assert_eq!(cache.len(), 1);

        assert!(cache.get(&"a".to_string()).is_none());
        assert!(cache.get(&"b".to_string()).is_some());
    }

    #[test]
    fn test_cache_thread_safety() {
        use std::thread;

        let cache = Arc::new(EmbeddingCache::new(100));
        let mut handles = vec![];

        // Spawn multiple threads inserting and reading
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let key = format!("key_{}_{}", i, j);
                    cache_clone.insert(key.clone(), vec![i as f32, j as f32]);
                    let _ = cache_clone.get(&key);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 100 entries (10 threads * 10 keys)
        assert_eq!(cache.len(), 100);
    }
}
