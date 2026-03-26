//! I/O module - SSD persistence for sector weights
//!
//! Two-file format per sector:
//!   `{base}.tensors`  - safetensors: encoder, memory_bank, value_bank, decoder, emotional, summary
//!   `{base}.meta`     - bincode:     SectorMetadata (slots_used, timestamps, scores, etc.)
//!
//! Rayon is used for parallel multi-sector loading at startup.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::{Device, Tensor};
use rayon::prelude::*;

use crate::sector::{SectorMetadata, SectorWeights};

// ── File naming ──────────────────────────────────────────────────────────────

fn tensor_path(base: &Path) -> PathBuf {
    base.with_extension("tensors")
}

fn meta_path(base: &Path) -> PathBuf {
    base.with_extension("meta")
}

// ── Save ─────────────────────────────────────────────────────────────────────

pub fn save_sector(weights: &SectorWeights, base: &Path) -> Result<()> {
    save_tensors(weights, base)
        .with_context(|| format!("Failed to save tensors to {:?}", tensor_path(base)))?;
    save_metadata(&weights.metadata, base)
        .with_context(|| format!("Failed to save metadata to {:?}", meta_path(base)))?;
    Ok(())
}

fn save_tensors(weights: &SectorWeights, base: &Path) -> Result<()> {
    if let Some(parent) = tensor_path(base).parent() {
        fs::create_dir_all(parent)?;
    }

    let tensors: HashMap<String, Tensor> = [
        ("encoder".to_string(),     weights.encoder.clone()),
        ("memory_bank".to_string(), weights.memory_bank.clone()),
        ("value_bank".to_string(),  weights.value_bank.clone()),
        ("decoder".to_string(),     weights.decoder.clone()),
        ("emotional".to_string(),   weights.emotional.clone()),
        ("summary".to_string(),     weights.summary.clone()),
    ]
    .into_iter()
    .collect();

    candle_core::safetensors::save(&tensors, tensor_path(base))
        .map_err(|e| anyhow!("safetensors save failed: {e}"))
}

fn save_metadata(metadata: &SectorMetadata, base: &Path) -> Result<()> {
    let path = meta_path(base);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let encoded = bincode::serialize(metadata)
        .context("bincode serialization of SectorMetadata failed")?;
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&encoded)?;
    Ok(())
}

// ── Load ─────────────────────────────────────────────────────────────────────

pub fn load_sector(base: &Path, device: &Device) -> Result<SectorWeights> {
    let tensors = load_tensors(base, device)
        .with_context(|| format!("Failed to load tensors from {:?}", tensor_path(base)))?;
    let metadata = load_metadata(base)
        .with_context(|| format!("Failed to load metadata from {:?}", meta_path(base)))?;
    Ok(tensors.into_sector_weights(metadata))
}

struct LoadedTensors {
    encoder:     Tensor,
    memory_bank: Tensor,
    value_bank:  Tensor,
    decoder:     Tensor,
    emotional:   Tensor,
    summary:     Tensor,
}

impl LoadedTensors {
    fn into_sector_weights(self, metadata: SectorMetadata) -> SectorWeights {
        SectorWeights {
            encoder:     self.encoder,
            memory_bank: self.memory_bank,
            value_bank:  self.value_bank,
            decoder:     self.decoder,
            emotional:   self.emotional,
            summary:     self.summary,
            metadata,
        }
    }
}

fn load_tensors(base: &Path, device: &Device) -> Result<LoadedTensors> {
    let path = tensor_path(base);
    let mut map = candle_core::safetensors::load(&path, device)
        .map_err(|e| anyhow!("safetensors load failed for {:?}: {e}", path))?;

    let take = |m: &mut HashMap<String, Tensor>, name: &str| -> Result<Tensor> {
        m.remove(name)
            .ok_or_else(|| anyhow!("Missing tensor '{name}' in {:?}", path))
    };

    Ok(LoadedTensors {
        encoder:     take(&mut map, "encoder")?,
        memory_bank: take(&mut map, "memory_bank")?,
        value_bank:  take(&mut map, "value_bank")?,
        decoder:     take(&mut map, "decoder")?,
        emotional:   take(&mut map, "emotional")?,
        summary:     take(&mut map, "summary")?,
    })
}

fn load_metadata(base: &Path) -> Result<SectorMetadata> {
    let path = meta_path(base);
    let mut file = File::open(&path)
        .with_context(|| format!("Cannot open metadata file {:?}", path))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    bincode::deserialize::<SectorMetadata>(&buf)
        .context("bincode deserialization of SectorMetadata failed")
}

// ── Parallel batch load ───────────────────────────────────────────────────────

pub fn load_sectors_parallel(bases: &[PathBuf], device: &Device) -> Vec<(PathBuf, SectorWeights)> {
    bases
        .par_iter()
        .filter_map(|base| {
            match load_sector(base, device) {
                Ok(weights) => Some((base.clone(), weights)),
                Err(e) => {
                    tracing::warn!("Skipping sector {:?}: {e}", base);
                    None
                }
            }
        })
        .collect()
}

// ── Directory scanning ────────────────────────────────────────────────────────

pub fn list_sectors(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut bases: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension()
                .map(|ext| ext == "meta")
                .unwrap_or(false)
        })
        .map(|e| e.path().with_extension(""))
        .collect();
    bases.sort();
    bases
}

#[allow(dead_code)]
pub fn sector_exists(base: &Path) -> bool {
    tensor_path(base).exists() && meta_path(base).exists()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_weights(device: &Device) -> SectorWeights {
        SectorWeights::new(64, 16, 32, 8, device)
            .expect("Failed to create test weights")
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("sector_test");
        let device = Device::Cpu;

        let original = test_weights(&device);
        save_sector(&original, &base).expect("save failed");
        assert!(sector_exists(&base));

        let loaded = load_sector(&base, &device).expect("load failed");

        assert_eq!(original.metadata.slots_used,  loaded.metadata.slots_used);
        assert_eq!(original.metadata.max_slots,   loaded.metadata.max_slots);
        assert_eq!(original.metadata.created_at,  loaded.metadata.created_at);
        assert_eq!(original.encoder.shape().dims(), loaded.encoder.shape().dims());
        assert_eq!(original.value_bank.shape().dims(), loaded.value_bank.shape().dims());
    }

    #[test]
    fn test_list_sectors() {
        let dir = tempdir().unwrap();
        let device = Device::Cpu;
        for i in 0..3 {
            let base = dir.path().join(format!("sector_{:04}", i));
            save_sector(&test_weights(&device), &base).unwrap();
        }
        assert_eq!(list_sectors(dir.path()).len(), 3);
    }

    #[test]
    fn test_parallel_load() {
        let dir = tempdir().unwrap();
        let device = Device::Cpu;
        let bases: Vec<PathBuf> = (0..4)
            .map(|i| {
                let base = dir.path().join(format!("sector_{:04}", i));
                save_sector(&test_weights(&device), &base).unwrap();
                base
            })
            .collect();
        assert_eq!(load_sectors_parallel(&bases, &device).len(), 4);
    }
}
