//! Sector weight tensors and forward pass logic
//!
//! Memory architecture: key-value attention
//!   memory_bank (keys):  [max_slots, rank]      — encoded patterns for routing
//!   value_bank  (values): [max_slots, field_dim] — raw patterns for retrieval
//!
//! During write:  encode(pattern) → memory_bank[slot],  pattern → value_bank[slot]
//! During retrieve: attn = softmax(query_low @ keys.T),  result = attn @ values
//!
//! This gives exact round-trip fidelity pre-training.
//! Post-training, encoder/decoder learn compressed representations,
//! and value_bank can be replaced by decoder projection.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use serde::{Serialize, Deserialize};

/// Core weights for a memory sector
#[derive(Debug)]
pub struct SectorWeights {
    /// Project query from field_dim to rank (attention routing)
    pub encoder: Tensor,        // [field_dim, rank]
    /// Memory bank — encoded keys for content-based attention
    pub memory_bank: Tensor,    // [max_slots, rank]
    /// Value bank — raw patterns returned on retrieval
    pub value_bank: Tensor,     // [max_slots, field_dim]
    /// Project retrieved memory back to field_dim (used post-training)
    pub decoder: Tensor,        // [rank, field_dim]
    /// Emotional modulation weights per slot
    pub emotional: Tensor,      // [max_slots, emotional_dims]
    /// Learned summary embedding for this sector
    pub summary: Tensor,        // [field_dim]
    /// Per-slot metadata
    pub metadata: SectorMetadata,
}

/// Non-trainable metadata per memory slot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectorMetadata {
    pub slots_used: usize,
    pub max_slots: usize,
    pub access_counts: Vec<u64>,
    pub timestamps: Vec<i64>,
    pub significance_scores: Vec<f32>,
    pub created_at: i64,
    pub frozen_at: Option<i64>,
}

impl SectorWeights {
    pub fn new(
        field_dim: usize,
        rank: usize,
        max_slots: usize,
        emotional_dims: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let encoder    = Tensor::randn(0.0f32, 0.02, &[field_dim, rank], device)?;
        let memory_bank = Tensor::zeros(&[max_slots, rank], DType::F32, device)?;
        let value_bank  = Tensor::zeros(&[max_slots, field_dim], DType::F32, device)?;
        let decoder    = Tensor::zeros(&[rank, field_dim], DType::F32, device)?;
        let emotional  = Tensor::zeros(&[max_slots, emotional_dims], DType::F32, device)?;
        let summary    = Tensor::zeros(&[field_dim], DType::F32, device)?;

        let now = chrono::Utc::now().timestamp();
        let metadata = SectorMetadata {
            slots_used: 0,
            max_slots,
            access_counts: vec![0u64; max_slots],
            timestamps: vec![0i64; max_slots],
            significance_scores: vec![0.0f32; max_slots],
            created_at: now,
            frozen_at: None,
        };

        Ok(Self { encoder, memory_bank, value_bank, decoder, emotional, summary, metadata })
    }

    /// Encode a query from field space to sector's low-rank space
    pub fn encode(&self, query: &Tensor) -> CandleResult<Tensor> {
        // [field_dim] -> [rank]
        query.unsqueeze(0)?.matmul(&self.encoder)?.squeeze(0)
    }

    /// Retrieve memories via content-based addressing.
    ///
    /// Uses memory_bank (encoded keys) for attention scores,
    /// value_bank (raw patterns) for the returned result.
    /// This gives exact round-trip fidelity regardless of encoder/decoder training state.
    pub fn retrieve(
        &self,
        query_low: &Tensor,
        emotional_state: Option<&Tensor>,
        _top_k: usize,
    ) -> CandleResult<Tensor> {
        let slots_used = self.metadata.slots_used;
        if slots_used == 0 {
            let field_dim = self.value_bank.dim(1)?;
            return Tensor::zeros(&[field_dim], DType::F32, query_low.device());
        }

        // Attention scores: query_low [rank] vs active keys [slots_used, rank]
        let active_keys = self.memory_bank.narrow(0, 0, slots_used)?;
        let scores = active_keys.matmul(&query_low.unsqueeze(1)?)?.squeeze(1)?;
        // scores: [slots_used]

        // Emotional modulation
        let modulated_scores = if let Some(emo) = emotional_state {
            let active_emo = self.emotional.narrow(0, 0, slots_used)?;
            let emo_bias = active_emo.matmul(&emo.unsqueeze(1)?)?.squeeze(1)?;
            (scores + emo_bias)?
        } else {
            scores
        };

        // Softmax attention weights
        let attn = candle_nn::ops::softmax(&modulated_scores, 0)?;
        // attn: [slots_used]

        // Weighted sum over value_bank (raw patterns) — exact retrieval pre-training
        let active_values = self.value_bank.narrow(0, 0, slots_used)?;
        // [1, slots_used] @ [slots_used, field_dim] -> [1, field_dim] -> [field_dim]
        attn.unsqueeze(0)?.matmul(&active_values)?.squeeze(0)
    }

    /// Write a new memory pattern into the next available slot.
    ///
    /// Writes encoded pattern to memory_bank (key) and raw pattern to value_bank (value).
    /// Candle tensors are immutable — slot writes use Vec round-trip.
    /// O(max_slots * dim) but only runs on significant moments, not every token.
    pub fn write_memory(
        &mut self,
        pattern: &Tensor,
        significance: f32,
        emotional_state: Option<&Tensor>,
    ) -> CandleResult<bool> {
        let slot = self.metadata.slots_used;
        if slot >= self.metadata.max_slots {
            return Ok(false);
        }

        let device    = pattern.device().clone();
        let rank      = self.encoder.dim(1)?;
        let field_dim = self.value_bank.dim(1)?;
        let max_slots = self.metadata.max_slots;

        // ── Write encoded key into memory_bank[slot] ──────────────────────
        let encoded_vec: Vec<f32> = self.encode(pattern)?.to_vec1()?;
        let mut bank_data: Vec<f32> = self.memory_bank.flatten_all()?.to_vec1()?;
        let key_start = slot * rank;
        bank_data[key_start..key_start + rank].copy_from_slice(&encoded_vec);
        self.memory_bank = Tensor::from_vec(bank_data, &[max_slots, rank], &device)?;

        // ── Write raw pattern into value_bank[slot] ───────────────────────
        let pattern_vec: Vec<f32> = pattern.flatten_all()?.to_vec1()?;
        let mut value_data: Vec<f32> = self.value_bank.flatten_all()?.to_vec1()?;
        let val_start = slot * field_dim;
        value_data[val_start..val_start + field_dim].copy_from_slice(&pattern_vec);
        self.value_bank = Tensor::from_vec(value_data, &[max_slots, field_dim], &device)?;

        // ── Write emotional state into emotional[slot] ────────────────────
        if let Some(emo) = emotional_state {
            let emotional_dims = self.emotional.dim(1)?;
            let emo_vec: Vec<f32> = emo.flatten_all()?.to_vec1()?;
            let mut emo_data: Vec<f32> = self.emotional.flatten_all()?.to_vec1()?;
            let emo_start = slot * emotional_dims;
            let copy_len = emo_vec.len().min(emotional_dims);
            emo_data[emo_start..emo_start + copy_len]
                .copy_from_slice(&emo_vec[..copy_len]);
            self.emotional = Tensor::from_vec(emo_data, &[max_slots, emotional_dims], &device)?;
        }

        // ── Update metadata ───────────────────────────────────────────────
        let now = chrono::Utc::now().timestamp();
        self.metadata.slots_used += 1;
        self.metadata.timestamps[slot] = now;
        self.metadata.significance_scores[slot] = significance;

        // Running average summary embedding
        let n = self.metadata.slots_used as f64;
        self.summary = ((&self.summary * ((n - 1.0) / n))? + &(pattern * (1.0 / n))?)?;

        Ok(true)
    }

    /// Capacity ratio (0.0 = empty, 1.0 = full)
    pub fn capacity_ratio(&self) -> f32 {
        self.metadata.slots_used as f32 / self.metadata.max_slots as f32
    }
}
