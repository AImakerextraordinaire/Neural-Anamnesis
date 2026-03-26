//! API module - REST interface for Python interop
//!
//! All tensor data crosses the wire as a TensorEnvelope:
//!   { "data": [f32, ...], "shape": [usize, ...] }
//!
//! Endpoints:
//!   POST /query     - Send field state -> retrieved memory tensor
//!   POST /write     - Write a memory pattern to the active sector
//!   GET  /status    - Full system state (sectors, cache, index)
//!   GET  /health    - Liveness check (for Python client health loop)

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::AppState;

// ── Wire format ───────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorEnvelope {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl TensorEnvelope {
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor, ApiError> {
        let expected_len: usize = self.shape.iter().product();
        if self.data.len() != expected_len {
            return Err(ApiError::bad_request(format!(
                "data length {} does not match shape {:?} (expected {})",
                self.data.len(), self.shape, expected_len
            )));
        }
        Tensor::from_vec(self.data.clone(), self.shape.as_slice(), device)
            .map_err(|e| ApiError::internal(format!("Tensor construction failed: {e}")))
    }

    pub fn from_tensor(t: &Tensor) -> Result<Self, ApiError> {
        let shape = t.shape().dims().to_vec();
        let data = t
            .flatten_all()
            .and_then(|f| f.to_vec1::<f32>())
            .map_err(|e| ApiError::internal(format!("Tensor serialization failed: {e}")))?;
        Ok(TensorEnvelope { data, shape })
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, message: msg.into() }
    }
    fn internal(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, message: msg.into() }
    }
    fn service_unavailable(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::SERVICE_UNAVAILABLE, message: msg.into() }
    }
    fn insufficient_storage(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::from_u16(507).unwrap(), message: msg.into() }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": self.message });
        (self.status, Json(body)).into_response()
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct QueryRequest {
    pub query: TensorEnvelope,
    pub emotional_state: Option<TensorEnvelope>,
    pub top_k: Option<usize>,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub memory: TensorEnvelope,
    pub sectors_used: Vec<String>,
    pub latency_us: u64,
}

#[derive(Deserialize)]
pub struct WriteRequest {
    pub pattern: TensorEnvelope,
    pub significance: f32,
    pub emotional_state: Option<TensorEnvelope>,
}

#[derive(Serialize)]
pub struct WriteResponse {
    pub success: bool,
    pub sector_id: String,
    pub slots_used: usize,
    pub capacity_ratio: f32,
    pub sector_allocated: bool,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub total_memories: usize,
    pub total_sectors: usize,
    pub active_sector: Option<SectorInfo>,
    pub archived_sectors: usize,
    pub cache_utilization: f32,
    pub index_sectors: usize,
    pub field_dim: usize,
}

#[derive(Serialize)]
pub struct SectorInfo {
    pub id: String,
    pub slots_used: usize,
    pub max_slots: usize,
    pub capacity_ratio: f32,
}

// ── Validation ────────────────────────────────────────────────────────────────

fn validate_field_dim(env: &TensorEnvelope, field_dim: usize, name: &str) -> Result<(), ApiError> {
    if env.shape != [field_dim] {
        return Err(ApiError::bad_request(format!(
            "'{name}' must have shape [{field_dim}], got {:?}", env.shape
        )));
    }
    Ok(())
}

fn validate_significance(sig: f32) -> Result<(), ApiError> {
    if !(0.0..=1.0).contains(&sig) {
        return Err(ApiError::bad_request(format!(
            "significance must be in [0.0, 1.0], got {sig}"
        )));
    }
    Ok(())
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "neural-anamnesis",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    let mgr   = state.sector_manager.read().await;
    let cache = state.cache.read().await;
    let index = state.index.read().await;

    let active_info = mgr.active.as_ref().map(|a| SectorInfo {
        id:             a.id.to_string(),
        slots_used:     a.weights.metadata.slots_used,
        max_slots:      a.weights.metadata.max_slots,
        capacity_ratio: a.weights.capacity_ratio(),
    });

    Json(StatusResponse {
        total_memories:    mgr.total_memories(),
        total_sectors:     1 + mgr.archives.len(),
        active_sector:     active_info,
        archived_sectors:  mgr.archives.len(),
        cache_utilization: cache.utilization(),
        index_sectors:     index.num_sectors(),
        field_dim:         state.config.field_dim,
    })
}

async fn query_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    let start     = Instant::now();
    let field_dim = state.config.field_dim;
    let top_k     = req.top_k.unwrap_or(state.config.top_k);
    let device    = Device::Cpu;  // AppState has no device field; all tensors live on CPU

    validate_field_dim(&req.query, field_dim, "query")?;

    let query_tensor = req.query.to_tensor(&device)?;
    let emo_tensor   = req.emotional_state
        .as_ref()
        .map(|e| e.to_tensor(&device))
        .transpose()?;

    let mgr         = state.sector_manager.read().await;
    let index       = state.index.read().await;
    let mut cache   = state.cache.write().await;

    // index.route() returns Vec<(usize, f32)> — archive positions + relevance scores
    let ranked = index.route(&query_tensor, top_k)
        .map_err(|e| ApiError::internal(format!("Index routing failed: {e}")))?;

    let mut retrieved_vecs: Vec<Tensor> = Vec::new();
    let mut sectors_used:   Vec<String> = Vec::new();

    for (archive_idx, _score) in &ranked {
        if let Some(archive) = mgr.archives.get(*archive_idx) {
            let result = archive
                .retrieve(&query_tensor, emo_tensor.as_ref(), top_k)
                .map_err(|e| ApiError::internal(format!(
                    "Retrieve from archive {} failed: {e}", archive.id
                )))?;
            cache.record_access(archive.id, *archive_idx, 0.0);
            retrieved_vecs.push(result);
            sectors_used.push(archive.id.to_string());
        }
    }

    // Always include the active sector
    if let Some(active) = &mgr.active {
        let result = active
            .retrieve(&query_tensor, emo_tensor.as_ref(), top_k)
            .map_err(|e| ApiError::internal(format!("Retrieve from active sector failed: {e}")))?;
        retrieved_vecs.push(result);
        sectors_used.push(format!("active:{}", active.id));
    }

    if retrieved_vecs.is_empty() {
        let zero = Tensor::zeros(&[field_dim], candle_core::DType::F32, &device)
            .map_err(|e| ApiError::internal(format!("Zero tensor failed: {e}")))?;
        return Ok(Json(QueryResponse {
            memory:       TensorEnvelope::from_tensor(&zero)?,
            sectors_used: Vec::new(),
            latency_us:   start.elapsed().as_micros() as u64,
        }));
    }

    let memory = Tensor::stack(&retrieved_vecs, 0)
        .and_then(|s| s.mean(0))
        .map_err(|e| ApiError::internal(format!("Aggregation failed: {e}")))?;

    Ok(Json(QueryResponse {
        memory: TensorEnvelope::from_tensor(&memory)?,
        sectors_used,
        latency_us: start.elapsed().as_micros() as u64,
    }))
}

async fn write_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WriteRequest>,
) -> Result<Json<WriteResponse>, ApiError> {
    let field_dim = state.config.field_dim;
    let device    = Device::Cpu;

    validate_field_dim(&req.pattern, field_dim, "pattern")?;
    validate_significance(req.significance)?;

    let pattern_tensor = req.pattern.to_tensor(&device)?;
    let emo_tensor     = req.emotional_state
        .as_ref()
        .map(|e| e.to_tensor(&device))
        .transpose()?;

    let mut mgr = state.sector_manager.write().await;

    if mgr.active.is_none() {
        mgr.allocate_sector()
            .map_err(|e| ApiError::insufficient_storage(format!(
                "Cannot allocate initial sector: {e}"
            )))?;
    }

    let sector_allocated = mgr.active.as_ref().unwrap().weights.capacity_ratio() >= 1.0;

    if sector_allocated {
        info!("Active sector full — freezing and allocating new sector");
        mgr.freeze_active()
            .map_err(|e| ApiError::internal(format!("Sector freeze failed: {e}")))?;
        mgr.allocate_sector()
            .map_err(|e| ApiError::insufficient_storage(format!(
                "New sector allocation failed: {e}"
            )))?;
    }

    let active = mgr.active.as_mut()
        .ok_or_else(|| ApiError::service_unavailable("No active sector available"))?;

    let written = active.weights
        .write_memory(&pattern_tensor, req.significance, emo_tensor.as_ref())
        .map_err(|e| ApiError::internal(format!("Write failed: {e}")))?;

    if !written {
        warn!("Sector reported full during write — race condition");
        return Err(ApiError::service_unavailable("Active sector filled mid-write"));
    }

    Ok(Json(WriteResponse {
        success:          true,
        sector_id:        active.id.to_string(),
        slots_used:       active.weights.metadata.slots_used,
        capacity_ratio:   active.weights.capacity_ratio(),
        sector_allocated,
    }))
}

// ── Router ────────────────────────────────────────────────────────────────────

pub async fn serve(state: Arc<AppState>, port: u16) {
    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/query",  post(query_memories))
        .route("/write",  post(write_memory))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    info!("Neural Anamnesis API listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await
        .expect("Failed to bind");
    axum::serve(listener, app).await
        .expect("Server error");
}