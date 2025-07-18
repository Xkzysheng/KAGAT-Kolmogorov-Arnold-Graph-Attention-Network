# KAGAT: Kolmogorov-Arnold Graph Attention Network

**Paper**: [Springer Link](https://link.springer.com/chapter/10.1007/978-981-96-9946-9_22) | [DOI](https://doi.org/10.1007/978-981-96-9946-9_22)

> **Abstract**
> The classic Graph Attention Network (GAT) generates embeddings via attention-based neighborhood aggregation. Recent studies show Kolmogorov-Arnold Network (KAN) outperforms Multi-Layer Perceptrons (MLP) in various tasks. This work integrates KAN into GAT, proposing five novel KAGAT variants. Node classification experiments demonstrate that KAGATs surpass classic GNNs (GCN, GAT, GATv2, GIN) and existing KAN-based GNNs.

## Model Architecture Variants
Repository contains sparse/dense implementations of:
1. **KAGAT-NA1** (`KAGAT-NA1.py`)
   Neighborhood aggregation → KAN layer (vanilla GAT attention)
2. **KAGAT-NA2** (`KAGAT-NA2.py`)
   KAN layer → Neighborhood aggregation (vanilla GAT attention)
3. **KAGAT-AT** (`KAGAT-AT.py`)
   KAN replaces MLP in attention computation (vanilla aggregation)
4. **KAGAT-NA1-AT** (`KAGAT-NA1-AT.py`)
   NA1 architecture + KAN-based attention
5. **KAGAT-NA2-AT** (`KAGAT-NA2-AT.py`)
   NA2 architecture + KAN-based attention

## Key Contributions
✅ Novel fusion of KAN (ICLR'25) and GAT (ICLR'18)
✅ State-of-the-art performance on node classification tasks
✅ Superiority over classic GNNs and KAN-GNNs baselines

## Citation
```bibtex
@InProceedings{Gong2025KAGAT,
  author    = {Gong, Haoran and An, Zhuojun and Mou, Jialong and Cheng, Jianjun and Liu, Li},
  title     = {KAGAT: Kolmogorov-Arnold Graph Attention Network},
  booktitle = {Advanced Intelligent Computing Technology and Applications (ICIC 2025)},
  publisher = {Springer},
  year      = {2025},
  volume    = {2565},
  pages     = {221--234},
  doi       = {10.1007/978-981-96-9946-9_22}
}
