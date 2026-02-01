# Parallel Matrix Multiplication with CPU + GPU Workers

[![CI](https://github.com/YOUR_USERNAME/matmul-workers/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/matmul-workers/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Rust system using the [Burn](https://burn.dev) framework to process matrix multiplications in parallel across CPU and GPU simultaneously.

## Architecture

```
                    ┌─────────────────┐
                    │   Dispatcher    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │ CPU-0   │    │ CPU-1   │    │   GPU   │
        │(ndarray)│    │(ndarray)│    │ (wgpu)  │
        └─────────┘    └─────────┘    └─────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Results      │
                    └─────────────────┘
```

- **CPU Workers**: Use `burn-ndarray` backend, one per core (half of available cores to avoid contention)
- **GPU Worker**: Single worker using `burn-wgpu` backend
- **Channels**: Bounded `crossbeam-channel` for job distribution and result collection

## Performance Characteristics

| Backend | First Job | Subsequent Jobs | Best For |
|---------|-----------|-----------------|----------|
| GPU     | ~200ms (shader compile) | ~2ms | Large matrices, batches |
| CPU     | ~7ms | ~7ms | Small matrices, consistent latency |

## Usage

```bash
cargo run --release
```

Example output:
```
Matrix multiplication workers demo
  CPU workers: 5
  Matrix size: 512x512
  Total jobs: 20

Job 1 completed by CPU-0 in 9.08ms (result[0]=127.7262)
Job 2 completed by CPU-1 in 8.26ms (result[0]=128.6880)
...
Job 0 completed by GPU in 214.76ms (result[0]=129.7924)  <- first GPU job (shader compile)
Job 3 completed by GPU in 1.80ms (result[0]=123.7188)    <- subsequent GPU jobs fast
...
Total time: 231.29ms
```

## Key Design Decisions

1. **Bounded channels** - Prevents memory blowup, provides backpressure when workers are saturated
2. **Half CPU cores** - Avoids contention with system and GPU driver threads
3. **Single GPU worker** - GPU operations are already parallel internally; multiple workers would just contend
4. **Job routing at submit** - Caller decides CPU vs GPU based on matrix size or other heuristics

## API

```rust
let dispatcher = Dispatcher::new(num_cpu_workers);

// Submit to CPU pool
dispatcher.submit_cpu(job);

// Submit to GPU
dispatcher.submit_gpu(job);

// Collect results
let result = dispatcher.results().recv().unwrap();

// Shutdown (finishes pending work)
dispatcher.shutdown();
```

## Dependencies

- `burn` - Deep learning framework with multiple backends
- `crossbeam-channel` - Fast multi-producer multi-consumer channels
- `num_cpus` - Detect available CPU cores
