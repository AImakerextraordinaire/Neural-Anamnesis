//! Sector lifecycle management - ownership-enforced state transitions
//!
//! ActiveSector: mutable, accepts new memories, lives in RAM
//! ArchivedSector: immutable, read-only, lives on SSD
//!
//! The freeze() method CONSUMES the ActiveSector and returns an ArchivedSector.
//! The type system makes it impossible to write to a frozen sector.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use candle_core::{Device, Tensor, Result as CandleResult};
use uuid::Uuid;
use tracing::info;
use anyhow::Result;

use super::SectorWeights;
use crate::NeuralAnamnConfig;
use crate::io;

/// An active sector that accepts new memory writes.
/// Lives in RAM. Mutable. Only one active sector at a time.
pub struct ActiveSector {
    pub id: Uuid,
    pub weights: SectorWeights,
    pub device: Device,
    /// Validation memory indices for interference detection
    #[allow(dead_code)]
    validation_indices: Vec<usize>,
}

/// A frozen sector. Read-only. Lives on SSD, cached in RAM.
/// Arc-wrapped weights enable zero-copy sharing across retrieval threads.
pub struct ArchivedSector {
    pub id: Uuid,
    pub weights: Arc<SectorWeights>,
    pub path: PathBuf,
}

impl ActiveSector {
    /// Create a new empty active sector
    pub fn new(config: &NeuralAnamnConfig) -> CandleResult<Self> {
        let device = Device::Cpu;
        let weights = SectorWeights::new(
            config.field_dim,
            config.rank,
            config.slots_per_sector,
            config.emotional_dims,
            &device,
        )?;

        Ok(Self {
            id: Uuid::new_v4(),
            weights,
            device,
            validation_indices: Vec::new(),
        })
    }

    /// Write a new memory to this sector
    pub fn write_memory(
        &mut self,
        pattern: &Tensor,
        significance: f32,
        emotional_state: Option<&Tensor>,
    ) -> CandleResult<bool> {
        self.weights.write_memory(pattern, significance, emotional_state)
    }

    /// Retrieve memories from this sector (active sectors are also readable)
    pub fn retrieve(
        &self,
        query: &Tensor,
        emotional_state: Option<&Tensor>,
        top_k: usize,
    ) -> CandleResult<Tensor> {
        let query_low = self.weights.encode(query)?;
        self.weights.retrieve(&query_low, emotional_state, top_k)
    }

    /// Check if this sector is approaching its interference threshold
    pub fn should_freeze(&self) -> bool {
        // Simple capacity-based check for now
        // Phase 5: Replace with validation-loss-based interference detection
        self.weights.capacity_ratio() > 0.85
    }

    /// Freeze this sector into an immutable archive.
    /// CONSUMES self - the active sector ceases to exist.
    /// Returns an ArchivedSector that can never be written to.
    /// Also persists weights to disk via io::save_sector.
    pub fn freeze(mut self, storage_dir: &Path) -> Result<ArchivedSector> {
        let now = chrono::Utc::now().timestamp();
        self.weights.metadata.frozen_at = Some(now);

        // Base path (no extension) — io module appends .tensors / .meta
        let base = storage_dir.join(format!("sector_{}", self.id));

        // Persist to SSD before releasing mutable ownership
        io::save_sector(&self.weights, &base)?;

        info!(
            "Sector {} frozen and saved: {}/{} slots used, base path: {}",
            self.id,
            self.weights.metadata.slots_used,
            self.weights.metadata.max_slots,
            base.display()
        );

        Ok(ArchivedSector {
            id:      self.id,
            weights: Arc::new(self.weights),
            path:    base,
        })
    }

    /// Get the summary embedding for index layer registration
    pub fn summary_embedding(&self) -> &Tensor {
        &self.weights.summary
    }

    /// Current capacity info
    pub fn status(&self) -> SectorStatus {
        SectorStatus {
            id:             self.id,
            slots_used:     self.weights.metadata.slots_used,
            max_slots:      self.weights.metadata.max_slots,
            capacity_ratio: self.weights.capacity_ratio(),
            is_frozen:      false,
        }
    }
}

impl ArchivedSector {
    /// Retrieve memories from this archived sector (read-only)
    pub fn retrieve(
        &self,
        query: &Tensor,
        emotional_state: Option<&Tensor>,
        top_k: usize,
    ) -> CandleResult<Tensor> {
        let query_low = self.weights.encode(query)?;
        self.weights.retrieve(&query_low, emotional_state, top_k)
    }

    /// Get the summary embedding for index layer
    pub fn summary_embedding(&self) -> &Tensor {
        &self.weights.summary
    }

    /// Status info
    pub fn status(&self) -> SectorStatus {
        SectorStatus {
            id:             self.id,
            slots_used:     self.weights.metadata.slots_used,
            max_slots:      self.weights.metadata.max_slots,
            capacity_ratio: self.weights.capacity_ratio(),
            is_frozen:      true,
        }
    }
}

/// Manages the collection of sectors (one active + N archived)
pub struct SectorManager {
    pub active:   Option<ActiveSector>,
    pub archives: Vec<ArchivedSector>,
    config:       NeuralAnamnConfig,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SectorStatus {
    pub id:             Uuid,
    pub slots_used:     usize,
    pub max_slots:      usize,
    pub capacity_ratio: f32,
    pub is_frozen:      bool,
}

impl SectorManager {
    pub fn new(config: &NeuralAnamnConfig) -> Self {
        let active = ActiveSector::new(config)
            .expect("Failed to create initial active sector");

        info!("SectorManager initialized with active sector {}", active.id);

        Self {
            active:   Some(active),
            archives: Vec::new(),
            config:   config.clone(),
        }
    }

    // ── Public API surface (used by api/mod.rs) ───────────────────────────

    /// Allocate a fresh active sector.
    /// Called by the API when no active sector exists yet, or after a manual freeze.
    pub fn allocate_sector(&mut self) -> Result<()> {
        let new_active = ActiveSector::new(&self.config)
            .map_err(|e| anyhow::anyhow!("Failed to create active sector: {e}"))?;
        info!("New active sector allocated: {}", new_active.id);
        self.active = Some(new_active);
        Ok(())
    }

    /// Freeze the active sector to disk WITHOUT allocating a new one.
    /// The API layer handles allocation separately so it controls
    /// the freeze→allocate sequence explicitly.
    pub fn freeze_active(&mut self) -> Result<()> {
        let active = self.active.take()
            .ok_or_else(|| anyhow::anyhow!("No active sector to freeze"))?;

        let archived = active.freeze(&self.config.storage_dir)?;
        info!(
            "Sector {} archived. Total archives: {}",
            archived.id,
            self.archives.len() + 1
        );
        self.archives.push(archived);
        Ok(())
    }

    /// Look up an archived sector by UUID string.
    /// Returns a reference to the Arc-wrapped weights if found.
    pub fn get_archived(&self, id: &Uuid) -> Option<Arc<SectorWeights>> {
        self.archives
            .iter()
            .find(|a| &a.id == id)
            .map(|a| Arc::clone(&a.weights))
    }

    // ── Internal write path (used by write_memory below) ─────────────────

    /// Write a memory to the active sector. Auto-freezes and allocates if needed.
    pub fn write_memory(
        &mut self,
        pattern: &Tensor,
        significance: f32,
        emotional_state: Option<&Tensor>,
    ) -> CandleResult<bool> {
        if let Some(ref active) = self.active {
            if active.should_freeze() {
                self.freeze_and_allocate()
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            }
        }

        if self.active.is_none() {
            self.allocate_sector()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }

        if let Some(ref mut active) = self.active {
            active.write_memory(pattern, significance, emotional_state)
        } else {
            Ok(false)
        }
    }

    /// Freeze the active sector and immediately allocate a fresh one.
    /// Used internally by the write path (auto-roll on capacity).
    fn freeze_and_allocate(&mut self) -> Result<()> {
        self.freeze_active()?;
        self.allocate_sector()
    }

    // ── Startup: load existing sectors from SSD ───────────────────────────

    /// Scan `storage_dir` for archived sectors and load them in parallel.
    /// Returns the number of sectors successfully loaded.
    pub fn load_archived_sectors(&mut self, storage_dir: &Path) -> usize {
        let bases = io::list_sectors(storage_dir);
        if bases.is_empty() {
            info!("No archived sectors found in {:?}", storage_dir);
            return 0;
        }

        info!("Loading {} archived sectors from {:?}", bases.len(), storage_dir);

        let device = Device::Cpu;
        let loaded = io::load_sectors_parallel(&bases, &device);
        let count  = loaded.len();

        for (base, weights) in loaded {
            // Reconstruct the UUID from the filename: sector_{uuid}
            let id = base
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_prefix("sector_"))
                .and_then(|n| Uuid::parse_str(n).ok())
                .unwrap_or_else(Uuid::new_v4);

            self.archives.push(ArchivedSector {
                id,
                weights: Arc::new(weights),
                path: base,
            });
        }

        info!("Loaded {} archived sectors", count);
        count
    }

    // ── Status ────────────────────────────────────────────────────────────

    pub fn status(&self) -> Vec<SectorStatus> {
        let mut statuses = Vec::new();
        if let Some(ref active) = self.active {
            statuses.push(active.status());
        }
        for archive in &self.archives {
            statuses.push(archive.status());
        }
        statuses
    }

    pub fn total_memories(&self) -> usize {
        let active_count = self.active.as_ref()
            .map(|a| a.weights.metadata.slots_used)
            .unwrap_or(0);
        let archive_count: usize = self.archives.iter()
            .map(|a| a.weights.metadata.slots_used)
            .sum();
        active_count + archive_count
    }
}
