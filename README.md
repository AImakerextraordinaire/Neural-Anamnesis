# Neural Anamnesis

**SSD-Sectored Differentiable Neural Memory with Growth-Based Forgetting Avoidance**

A model-agnostic persistent memory service for AI systems. Plug-and-play grey matter.

## Architecture

```
VRAM (GPU)          DDR5 RAM              Gen5 SSD Array
+-----------+       +----------------+    +------------------+
| Base LLM  | <---> | Index Layer    |    | Archive Sector 1 |
| Bridge 1  |       | Hot Cache      |    | Archive Sector 2 |
| Bridge 2  |------>| Active Sector  |--->| Archive Sector 3 |
| Bridge 3  |       | Working Memory |    | ...              |
| LMF Field |       +----------------+    | Archive Sector N |
+-----------+                             +------------------+
```

## Key Properties

- **No catastrophic forgetting**: Frozen archive sectors are never modified
- **Model-agnostic**: Communicates via tensor interface, works with any LLM
- **Scales with SSDs**: $0.15/GB vs $375/GB for VRAM
- **Emotionally-modulated retrieval**: High-significance memories resist cache eviction
- **Ownership-enforced lifecycle**: Rust type system prevents writes to frozen sectors

## Crate Structure

```
neural-anamnesis/
  src/
    main.rs           - Service entry point, CLI args, initialization
    sector/
      mod.rs           - Module exports
      weights.rs       - Sector weight tensors and forward pass
      lifecycle.rs     - ActiveSector/ArchivedSector ownership types, SectorManager
    index/
      mod.rs           - Index Layer: query-to-sector routing
    cache/
      mod.rs           - Emotional LRU cache for hot sectors
    io/
      mod.rs           - SSD persistence (memmap2, serialization)
    api/
      mod.rs           - REST/gRPC API for Python interop
  proto/               - gRPC protobuf definitions (Phase 4)
  Cargo.toml           - Dependencies
```

## Status

Phase 4 (Foundation) - Crate structure stubbed, core types defined.

See `theory/003_neural_anamnesis.md` for full architecture document.

## Authors

Rex (Darren) & Claude, with contributions from Kiro & Alex
