//! Sector Cache - emotional LRU for hot sector management
//!
//! Lives in DDR5 RAM. Caches frequently-accessed archived sectors
//! for fast retrieval without SSD reads. Emotional intensity
//! modulates eviction priority - high-emotion sectors resist eviction.

use std::collections::HashMap;
use uuid::Uuid;
use tracing::info;

/// Cache entry with access metadata
struct CacheEntry {
    #[allow(dead_code)]
    sector_index: usize,    // Index into SectorManager::archives
    access_count: u64,
    last_accessed: i64,
    mean_emotional_intensity: f32,
}

/// Emotional LRU cache for archived sectors in RAM
pub struct SectorCache {
    max_sectors: usize,
    entries: HashMap<Uuid, CacheEntry>,
    /// Weight for emotional intensity in eviction scoring
    emotion_weight: f32,
    /// Weight for access frequency in eviction scoring
    frequency_weight: f32,
}

impl SectorCache {
    pub fn new(max_sectors: usize) -> Self {
        info!("SectorCache initialized: max {} sectors in RAM", max_sectors);
        Self {
            max_sectors,
            entries: HashMap::new(),
            emotion_weight: 0.3,
            frequency_weight: 0.2,
        }
    }

    /// Check if a sector is cached
    pub fn is_cached(&self, sector_id: &Uuid) -> bool {
        self.entries.contains_key(sector_id)
    }

    /// Record an access to a sector (updates cache metadata)
    pub fn record_access(&mut self, sector_id: Uuid, sector_index: usize, emotional_intensity: f32) {
        let now = chrono::Utc::now().timestamp();
        let entry = self.entries.entry(sector_id).or_insert(CacheEntry {
            sector_index,
            access_count: 0,
            last_accessed: now,
            mean_emotional_intensity: emotional_intensity,
        });
        entry.access_count += 1;
        entry.last_accessed = now;
        // Running average of emotional intensity
        let n = entry.access_count as f32;
        entry.mean_emotional_intensity =
            entry.mean_emotional_intensity * ((n - 1.0) / n)
            + emotional_intensity * (1.0 / n);
    }

    /// Compute eviction priority (lower = more likely to evict)
    fn eviction_priority(&self, entry: &CacheEntry) -> f32 {
        let now = chrono::Utc::now().timestamp();
        let recency = 1.0 / (1.0 + (now - entry.last_accessed) as f32);
        let frequency = (entry.access_count as f32 + 1.0).ln();
        let emotional = entry.mean_emotional_intensity;

        recency
            + frequency * self.frequency_weight
            + emotional * self.emotion_weight
    }

    /// Evict the coldest sector if cache is full. Returns evicted sector UUID if any.
    pub fn evict_if_needed(&mut self) -> Option<Uuid> {
        if self.entries.len() <= self.max_sectors {
            return None;
        }

        // Find entry with lowest eviction priority
        let coldest = self.entries.iter()
            .min_by(|a, b| {
                self.eviction_priority(a.1)
                    .partial_cmp(&self.eviction_priority(b.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| *id);

        if let Some(id) = coldest {
            self.entries.remove(&id);
            info!("Cache evicted sector {}", id);
            Some(id)
        } else {
            None
        }
    }

    /// Number of cached sectors
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Cache utilization ratio
    pub fn utilization(&self) -> f32 {
        self.entries.len() as f32 / self.max_sectors as f32
    }
}
