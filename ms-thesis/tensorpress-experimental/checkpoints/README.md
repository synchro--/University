# Checkpoints

This folder holds the consolidated PyTorch checkpoints for the main model families used in this project.

## Layout

| File | Model family | Original source (most recent keeper) |
|------|--------------|--------------------------------------|
| `nin.pth` | Network in Network (NIN) | `NIN/NIN.conv7.dump.pth` |
| `zhang.pth` | Zhang et al. CNN | `zhang/cp/m_cp-reverse.conv3.pth` |
| `lenet.pth` | LeNet-4 on CIFAR | `saved_models/lenet4/cp-conv1.pth` |

## Consolidation (2026-06-11)

The following cleanup was applied:

1. **Merged scattered checkpoints** from `saved_models/`, `NIN/`, `zhang/`, `models/`, and `checkpoints/` into this single `checkpoints/` directory. The old `saved_models/` tree was removed.
2. **One checkpoint per model family** — for each of NIN, Zhang, and LeNet, only the most recent `.pth` file was kept (by modification time, then name hints such as `best` / `last` / `full`, then file size). All other `.pth` duplicates for that family were deleted.
3. **Removed misc experiment checkpoints** — intermediate states, layer dumps, finetune artifacts, and other non-family checkpoints (e.g. `states-*`, `models-*`, `classic.pth`, `finetuned.pth`) were deleted. Only the three family checkpoints above remain.

## Compression

The folder can be archived with [pixz](https://github.com/vasi/pixz) (parallel, indexed XZ) at maximum ratio:

```bash
cd /path/to/cnn-decomposer-tensorpress-migration
tar --use-compress-program="pixz -9 -e" -cf checkpoints.tar.tpxz checkpoints/
```

- `-9` — highest xz compression level
- `-e` — extreme mode (slowest, slightly smaller output)

On macOS, prefer `--use-compress-program` over `tar -I"pixz …"` because BSD `tar` does not accept a multi-argument compress program via `-I`.

### Decompress

```bash
cd /path/to/cnn-decomposer-tensorpress-migration
tar --use-compress-program=pixz -xf checkpoints.tar.tpxz
```

This recreates the `checkpoints/` directory with `nin.pth`, `zhang.pth`, `lenet.pth`, and this README.

### List archive contents (without extracting)

```bash
pixz -l checkpoints.tar.tpxz
```

## Loading a checkpoint

```python
import torch

model = torch.load("checkpoints/nin.pth", map_location="cpu", weights_only=False)
```

Adjust `weights_only` and loading logic to match how each checkpoint was saved (full model vs state dict).
