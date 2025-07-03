# Generative AI-enabled Wireless Communications for Robust Low-Altitude Economy Networking

This repository implements a **diffusion model-based optimization framework** for wireless communications in Low-Altitude Economy Networks (LAENets), leveraging a Mixture-of-Experts (MoE) Transformer to enhance robustness under uncertainty.

---

## 📂 Directory Structure

- `diffusion/model.py`  
  Contains the **diffusion model** and **MoE-Transformer actor network** implementation.

- `env/env_uav.py`  
  Defines the **UAV simulation environment**, including uncertain channels and system dynamics.

- `MoE_trans_GDM.py`  
  Entry script to **run training and testing** for the proposed algorithm under various optimization settings.

- `.results/`  
  Stores **output logs, model checkpoints, and evaluation results** from each experiment.

---

## ▶️ Run the Code

```bash
python MoE_trans_GDM.py

---

## 🔧 Based On

This project builds upon the [GDMOPT repository](https://github.com/HongyangDu/GDMOPT) by Hongyang Du, with additional enhancements for MoE-based, uncertainty-aware optimization in diffusion-driven wireless systems.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{zhao2025generative,
  title={Generative AI-enabled wireless communications for robust low-altitude economy networking},
  author={Zhao, Changyuan and Wang, Jiacheng and Zhang, Ruichen and Niyato, Dusit and Sun, Geng and Du, Hongyang and Kim, Dong In and Jamalipour, Abbas},
  journal={arXiv preprint arXiv:2502.18118},
  year={2025}
}

