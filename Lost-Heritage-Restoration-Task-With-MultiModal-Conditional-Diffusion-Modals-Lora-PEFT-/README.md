Monument & Statue Restoration with Fine-Tuned Stable Diffusion XL

This project aims to restore damaged monuments and statues with high realism using a fine-tuned Stable Diffusion XL model. It leverages LoRA (Low-Rank Adaptation), PEFT (Parameter-Efficient Fine-Tuning), and conditioning on depth maps and captions for optimal performance on 1024×1024 images.

---

📊 Project Overview

- Goal: Generate realistic restorations of damaged cultural heritage assets.
- Model: Fine-tuned **Stable Diffusion XL** with **LoRA + PEFT**.
- Image Resolution: 1024×1024
- Inputs:
  - Original image captions
  - Depth maps
  - Damaged versions of the original images
- Output: Fully restored high-fidelity monument/statue image

---

🗂️ Dataset Preparation

📦 Data Sources
- Monuments:
  - Kaggle public datasets
  - Roboflow framework datasets
- Statues:
  - Wikimedia Commons
  - APIs from Europeana, Smithsonian Museum, and MIT Museum Collections
🛠️ Preprocessing Pipeline
1. De-duplication and similarity filtering
2. Super-resolution using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (4×)
3. Denoising with [Waifu2x](https://github.com/nagadomi/waifu2x)
4. Depth Map Generation using [DPT Hybrid (MiDaS)](https://github.com/isl-org/MiDaS)
5. Captioning** using the [Joy-Transformer Caption Alpha-2](https://huggingface.co](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two)/) model
6. Feature Structuring:
   - Heritage type
   - Architectural style
   - Texture
   - Era of construction
   - Key features
   - Primary materials
   - Lighting conditions

---

🔧 Damage Simulation & Dataset Pairing

- Used intact images to generate four+ different types of realistically simulated damages, including:
  - Cracks and weathering
  - Large missing parts
  - Dirt and discoloration
  - Vandalism-like effects
- Resulted in a paired dataset of 40,000 image triplets:
  - Damaged image
  - Depth map
  - Caption + structural features

---

🧠 Model Training

- Base: Stable Diffusion XL-Inpainting
- Fine-Tuning:
  - Used LoRA and PEFT for efficient parameter tuning
  - Conditioned on dynamically generated **masked depth maps**
- Training Enhancements:
  - On-the-fly depth + dynamic damage mask generation
  - Improved time and GPU efficiency
  - Better generalization for unseen restoration patterns

---

📈 Results

The fine-tuned model exhibits:
- High fidelity in restoration outputs
- Strong texture and structural accuracy
- Realistic re-creation of missing parts and damaged areas
- Robustness to a wide variety of input damages

---

📜 License

This project is for research and educational purposes. Some datasets used are under respective creative commons and institutional licenses (e.g., Europeana, Wikimedia Commons).

---
🤝 Acknowledgements

Thanks to the developers of:
- [Stable Diffusion XL](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
- [LoRA & PEFT (HuggingFace)](https://huggingface.co/docs/peft/index)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [Waifu2x](https://github.com/nagadomi/waifu2x)
- [MiDaS (DPT Hybrid)](https://github.com/isl-org/MiDaS)
- [Joy Transformer Captioning](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two)
---

📬 Contact

For questions or collaboration or if you need access to the data used in this project: [Youssef-Mejdi] - [yousefmejdi5inscription@gmail.com] or open an issue in this repo.
