# 🚀 KAGAT: Kolmogorov-Arnold Graph Attention Network

> **ICIC 2025 Accepted Paper (Oral)**
>   
> **This repository includes the defination of our proposed all five KAGAT variants in the paper:  
> KAGAT-NA1, KAGAT-NA2, KAGAT-AT, KAGAT-NA1-AT and KAGAT-NA2-AT.**
>   
> **Full Paper Available at**: https://link.springer.com/chapter/10.1007/978-981-96-9946-9_22
## 🔍 Abstract
We introduce **KAGAT** - a novel fusion of Kolmogorov-Arnold Networks (KAN) with Graph Attention Networks (GAT). By leveraging KAN's superior high-dimensional learning capabilities and GAT's attention-based neighborhood aggregation, we propose **five innovative architectures** that consistently outperform classic GNNs (GCN, GAT, GATv2, GIN) and existing KAN-based GNNs on node classification tasks across multiple datasets.

## 🌟 Key Contributions
1. **Novel Hybrid Framework**
   First integration of KAN (ICLR'25) with GAT (ICLR'18)
2. **Five Specialized Variants**
   Systematic exploration of KAN integration strategies:
   - Neighborhood aggregation order (NA1/NA2)
   - Attention computation enhancement (AT)
   - Combined approaches (NA1-AT/NA2-AT)
3. **State-of-the-Art Performance**
   Outperforms 10+ baselines on 5 benchmark datasets
4. **Architecture Insights**
   Reveals KAN's intrinsic superiority over parameter scaling effects

## 🧠 Model Variants
| Variant              | Code File              | Architecture Summary                          |
|----------------------|------------------------|----------------------------------------------|
| **KAGAT-NA1**        | `KAGAT-NA1.py`         | Aggregation → KAN layer (vanilla attention)  |
| **KAGAT-NA2**        | `KAGAT-NA2.py`         | KAN layer → Aggregation + tanh activation    |
| **KAGAT-AT**         | `KAGAT-AT.py`          | KAN replaces MLP in attention computation    |
| **KAGAT-NA1-AT**     | `KAGAT-NA1-AT.py`      | NA1 + KAN-based attention                    |
| **KAGAT-NA2-AT**     | `KAGAT-NA2-AT.py`      | NA2 + KAN-based attention                    |

## 📊 Experimental Highlights
**Datasets**: Cora, CiteSeer, PubMed, Amazon-Photo, Amazon-Computers
**Key Results**:
- 🥇 **Best performance** on 4/5 datasets
- 📈 **Average accuracy gain**: +1.5% over vanilla GAT
- 💡 **KAN synergy**: Combined NA+AT variants outperform single-component integrations
- ⚖️ **Parameter efficiency**: Superiority persists even with parameter alignment  
## Our KAN Layer is consist with this B-spline based KAN: https://github.com/Blealtan/efficient-kan
