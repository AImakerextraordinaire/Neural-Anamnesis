//! Index Layer - lightweight neural router for sector discovery
//!
//! Lives in DDR5 RAM. Maps query tensors to sector relevance scores.
//! Uses learned sector summary embeddings for content-based routing.

use candle_core::{Tensor, Device, Result as CandleResult};
use tracing::info;

/// Neural router that maps queries to relevant sectors
pub struct IndexLayer {
    /// Summary embeddings for all registered sectors
    /// Shape: [num_sectors, hidden_dim]
    sector_summaries: Vec<Tensor>,
    /// Projection from field_dim to routing space
    query_proj_weight: Tensor,
    #[allow(dead_code)]
    hidden_dim: usize,
}

impl IndexLayer {
    pub fn new(field_dim: usize, hidden_dim: usize) -> Self {
        let device = Device::Cpu;
        let query_proj_weight = Tensor::randn(
            0.0f32, 0.02, &[field_dim, hidden_dim], &device,
        ).expect("Failed to init query projection");

        info!("IndexLayer initialized: field_dim={}, hidden_dim={}", field_dim, hidden_dim);

        Self {
            sector_summaries: Vec::new(),
            query_proj_weight,
            hidden_dim,
        }
    }

    /// Register a sector's summary embedding
    pub fn register_sector(&mut self, summary: Tensor) {
        self.sector_summaries.push(summary);
        info!("IndexLayer: {} sectors registered", self.sector_summaries.len());
    }

    /// Route a query to the top-K most relevant sectors
    /// Returns (sector_indices, relevance_scores)
    pub fn route(&self, query: &Tensor, top_k: usize) -> CandleResult<Vec<(usize, f32)>> {
        if self.sector_summaries.is_empty() {
            return Ok(Vec::new());
        }

        // Project query to routing space
        let q = query.unsqueeze(0)?.matmul(&self.query_proj_weight)?.squeeze(0)?;

        // Compute similarity with all sector summaries
        let mut scores: Vec<(usize, f32)> = Vec::new();
        for (i, summary) in self.sector_summaries.iter().enumerate() {
            // Dot product similarity
            let score = (&q * summary)?.sum_all()?.to_scalar::<f32>()?;
            scores.push((i, score));
        }

        // Sort by score descending, take top-K
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(scores)
    }

    /// Number of registered sectors
    pub fn num_sectors(&self) -> usize {
        self.sector_summaries.len()
    }
}
