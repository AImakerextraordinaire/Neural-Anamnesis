//! Sector module - self-contained differentiable memory units
//!
//! Each sector contains:
//!   - Encoder/decoder projections (field_dim <-> rank)
//!   - Memory bank (S slots x rank dimensions)
//!   - Emotional modulation weights
//!   - Metadata (timestamps, access counts, significance)
//!   - Summary embedding for index layer routing

mod weights;
mod lifecycle;

pub use weights::{SectorWeights, SectorMetadata};
#[allow(unused_imports)]
pub use lifecycle::{ActiveSector, ArchivedSector, SectorManager};