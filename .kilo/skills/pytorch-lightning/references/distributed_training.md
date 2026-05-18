# Distributed Training Reference

> **Load when**: training large models, using multiple GPUs, or needing memory-efficient strategies.

## Strategy Selection Decision Tree

```
Model fits on 1 GPU?
├── Yes → Single GPU (devices=1)
└── No → Model > GPU memory?
    ├── < 500M params → DDP (devices=4, strategy="ddp")
    ├── 500M - 10B → FSDP (devices=8, strategy="fsdp")
    └── > 10B → DeepSpeed ZeRO-3
```

## Strategy Comparison

| Strategy | Memory | Speed | Complexity | Best For |
|----------|--------|-------|------------|----------|
| DDP | Per-GPU copy | Fastest | Simple | < 500M params |
| FSDP | Sharded params | Fast | Medium | 500M+ params |
| DeepSpeed ZeRO-2 | Sharded optimizer + grads | Fast | Medium | Large models |
| DeepSpeed ZeRO-3 | Sharded params + optimizer + grads | Slower | High | Very large models |
| DeepSpeed ZeRO-1 | Sharded optimizer only | Fast | Simple | Memory-limited |

## DDP (Distributed Data Parallel)

The default strategy. Each GPU has a full model copy. Gradients are synced via all-reduce.

```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=4,
    strategy="ddp",                    # Explicit DDP
    # strategy="ddp_find_unused_parameters_true",  # If model has unused params
)

# Key settings
# DDP works automatically with self.log(..., sync_dist=True)
```

**Limitations**:
- Every GPU must fit the full model + optimizer + activations
- Gradient sync overhead increases with GPU count

## FSDP (Fully Sharded Data Parallel)

Shards model parameters across GPUs. Each GPU only holds the parameters it needs for its current computation.

```python
from lightning.pytorch.strategies import FSDPStrategy

trainer = L.Trainer(
    accelerator="gpu",
    devices=8,
    strategy=FSDPStrategy(
        sharding_strategy="FULL_SHARD",       # Shard params, grads, optimizer
        cpu_offload=False,                    # Offload to CPU (slower, more memory)
        auto_wrap_policy={nn.TransformerEncoderLayer},  # Wrap transformer layers
        activation_checkpointing_policy={nn.TransformerEncoderLayer},  # Trade compute for memory
    ),
)
```

**Sharding Strategies**:
| Strategy | What's Sharded | Memory Saving |
|----------|---------------|---------------|
| `FULL_SHARD` | Params + Grads + Optimizer | Maximum |
| `SHARD_GRAD_OP` | Grads + Optimizer | Medium |
| `HYBRID_SHARD` | Across nodes only | Good for multi-node |
| `NO_SHARD` | Nothing (DDP equivalent) | None |

**When to use FSDP**: Models with 500M+ parameters. For models under 500M, DDP is simpler and faster.

## DeepSpeed

Microsoft's optimization library. PyTorch Lightning supports it natively.

### ZeRO-2 (Sharded Optimizer + Gradients)

```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=8,
    strategy="deepspeed_stage_2",
    precision="16-mixed",
)
```

### ZeRO-3 (Sharded Everything)

```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=8,
    strategy="deepspeed_stage_3",
    precision="16-mixed",
)
```

### DeepSpeed Config File (Advanced)

```python
from lightning.pytorch.strategies import DeepSpeedStrategy

deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "train_batch_size": 256,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": True},
}

trainer = L.Trainer(
    strategy=DeepSpeedStrategy(config=deepspeed_config),
    devices=8,
)
```

## Communication Backends

| Backend | When |
|---------|------|
| `nccl` (default) | NVIDIA GPUs |
| `gloo` | CPU or cross-platform |
| `mpi` | HPC clusters |

## Common Patterns

### Mixed Precision + Distributed
```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    precision="16-mixed",  # Combined with DDP
)
```

### Log Aggregation in DDP
```python
def validation_step(self, batch, batch_idx):
    acc = self.compute_accuracy(batch)
    self.log("val/acc", acc, sync_dist=True)  # Averages across GPUs
```

### Find Unused Parameters
```python
# If your model has conditional layers (e.g., dynamic routing)
trainer = L.Trainer(
    strategy="ddp_find_unused_parameters_true"  # Slower, for conditional models
)
```

## Performance Tuning

| Issue | Fix |
|-------|-----|
| GPU utilization < 80% | Increase `num_workers`, use `pin_memory=True` |
| OOM on single GPU | Enable FSDP or gradient accumulation |
| Slow multi-GPU | Check NCCL is used, reduce `sync_dist` calls |
| Uneven GPU memory | Enable `ddp_find_unused_parameters_true` |
| Multi-node slow | FSDP with HYBRID_SHARD |

## Common Pitfalls

1. **`self.log()` without `sync_dist=True`** → Validation metrics wrong in multi-GPU
2. **Batch normalization in DDP** → BN stats are per-GPU, not global. Use SyncBatchNorm
3. **Random seeds not set** → Different GPUs may diverge: use `L.seed_everything(42)`
4. **Data loading bottleneck** → Increase workers, enable prefetch
5. **Gradient sync overhead** → Use `gradient_accumulation` to reduce sync frequency
