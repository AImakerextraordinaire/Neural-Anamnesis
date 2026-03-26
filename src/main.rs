//! Neural Anamnesis - SSD-Sectored Differentiable Neural Memory
//!
//! A three-tier memory service (VRAM / DDR5 RAM / Gen5 SSD) that provides
//! model-agnostic persistent memory with growth-based forgetting avoidance.
//!
//! Architecture:
//!   - Sectors: Self-contained differentiable memory modules (encode/store/decode)
//!   - Index Layer: Lightweight neural router for query-to-sector mapping
//!   - Cache: Emotional LRU for hot sector management in RAM
//!   - I/O: Memory-mapped SSD access for archived sectors
//!   - API: gRPC/REST interface for Python (PyTorch) interop

mod sector;
mod index;
mod cache;
mod io;
mod api;

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::sector::SectorManager;
use crate::index::IndexLayer;
use crate::cache::SectorCache;

/// Neural Anamnesis service configuration
#[derive(Parser, Debug)]
#[command(name = "neural-anamnesis")]
#[command(about = "SSD-sectored differentiable neural memory service")]
struct Args {
    /// Port for REST/gRPC API
    #[arg(short, long, default_value = "6060")]
    port: u16,

    /// Directory for sector storage (SSD)
    #[arg(short, long)]
    storage_dir: Option<PathBuf>,

    /// Field dimension (must match model hidden_dim)
    #[arg(long, default_value = "2880")]
    field_dim: usize,

    /// Sector compression rank
    #[arg(long, default_value = "64")]
    rank: usize,

    /// Memory slots per sector
    #[arg(long, default_value = "10000")]
    slots_per_sector: usize,

    /// Emotional dimensions
    #[arg(long, default_value = "17")]
    emotional_dims: usize,

    /// Max sectors to cache in RAM
    #[arg(long, default_value = "50")]
    cache_size: usize,

    /// Top-K sectors to retrieve per query
    #[arg(long, default_value = "5")]
    top_k: usize,
}

/// Shared application state
pub struct AppState {
    pub sector_manager: RwLock<SectorManager>,
    pub index: RwLock<IndexLayer>,
    pub cache: RwLock<SectorCache>,
    pub config: NeuralAnamnConfig,
}

/// Runtime configuration
#[derive(Debug, Clone)]
pub struct NeuralAnamnConfig {
    pub field_dim: usize,
    pub rank: usize,
    pub slots_per_sector: usize,
    pub emotional_dims: usize,
    pub cache_size: usize,
    pub top_k: usize,
    pub storage_dir: PathBuf,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let storage_dir = args.storage_dir
        .unwrap_or_else(|| home.join(".neural_anamnesis").join("sectors"));

    std::fs::create_dir_all(&storage_dir).expect("Failed to create storage directory");

    let config = NeuralAnamnConfig {
        field_dim: args.field_dim,
        rank: args.rank,
        slots_per_sector: args.slots_per_sector,
        emotional_dims: args.emotional_dims,
        cache_size: args.cache_size,
        top_k: args.top_k,
        storage_dir: storage_dir.clone(),
    };

    info!("=== Neural Anamnesis ===");
    info!("  Field dim: {}", config.field_dim);
    info!("  Rank: {}", config.rank);
    info!("  Slots/sector: {}", config.slots_per_sector);
    info!("  Emotional dims: {}", config.emotional_dims);
    info!("  Cache size: {} sectors", config.cache_size);
    info!("  Top-K retrieval: {}", config.top_k);
    info!("  Storage: {}", storage_dir.display());

    // Initialize components
    let sector_manager = SectorManager::new(&config);
    let index = IndexLayer::new(config.field_dim, config.rank);
    let cache = SectorCache::new(config.cache_size);

    let state = Arc::new(AppState {
        sector_manager: RwLock::new(sector_manager),
        index: RwLock::new(index),
        cache: RwLock::new(cache),
        config: config.clone(),
    });

    // Load existing sectors from disk
    {
        let mut mgr = state.sector_manager.write().await;
        let loaded = mgr.load_archived_sectors(&storage_dir);
        info!("  Loaded {} archived sectors from disk", loaded);
    }

    // Start API server
    info!("  Listening on port {}", args.port);
    api::serve(state, args.port).await;
}
