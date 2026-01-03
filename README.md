# 🤖 P-ML: An End-to-End AutoML Framework for Deploying Classical Machine Learning Models on Resource-Constrained Devices

<div align="center">

[![License](https://img.shields.io/badge/License-Academic_Use_Only-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Arduino%20%7C%20ESP32-green.svg)](README.md)
[![Python](https://img.shields.io/badge/Python-3.7%2B-yellow.svg)](README.md)
[![DOI](https://img.shields.io/badge/DOI-10.3897%2Fjucs-orange.svg)](https://doi.org/10.5281/zenodo.18130146)

**By Huu-Phuoc Nguyen**  
Can Tho University, Can Tho, Vietnam

[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--5822--4121-green.svg)](https://orcid.org/0009-0006-5822-4121) • [![Email](https://img.shields.io/badge/Email-huuphuoc.research%40gmail.com-red.svg)](mailto:huuphuoc.research@gmail.com)

</div>

---

## 📝 Abstract

This study presents **P-ML**, an end-to-end AutoML framework for deploying classical machine learning models on memory-constrained microcontrollers. The framework automates the complete workflow, including data splitting, model selection, hyperparameter optimization, and generation of optimized Arduino-compatible C++ libraries. P-ML integrates Optuna-based hyperparameter tuning with stratified data splitting methods such as SPXY and K-Fold cross-validation to ensure robust and reliable model selection. The generated libraries are compact, efficient, and directly deployable on Arduino Uno, Nano, and ESP32 platforms using the Arduino IDE. Experimental results demonstrate that P-ML enables accurate sensor data classification, achieving over 90% accuracy while maintaining a small memory footprint suitable for embedded IoT applications.

**Keywords:** AutoML, Embedded Machine Learning, Resource-Constrained Systems, Classical ML, Hyperparameter Tuning, Arduino Code Generation, TinyML

**Categories:** B.4.2 (Embedded Systems), C.3.3 (Real-time and Embedded Systems), I.2.6 (Learning)

**DOI:** [https://doi.org/10.5281/zenodo.18130146]

See the English user manual here: [link](GuideAutoML-Eng.pdf)

See the Vietnam user manual here: [link](GuideAutoML-Vi.pdf)

---

## 🚀 Introduction

The increasing deployment of Internet of Things (IoT) systems has driven a strong demand for machine learning inference directly on microcontroller-based devices such as Arduino and ESP32. These platforms are widely used in sensor networks, wearable systems, and edge intelligence applications, yet they operate under strict memory and computational constraints. In such environments, classical machine learning models—such as Support Vector Machines, decision trees, ensemble methods, and k-nearest neighbors—often outperform deep learning approaches for small and medium-sized sensor datasets while requiring significantly fewer resources.

Despite this advantage, deploying classical machine learning models on microcontrollers remains challenging. Existing embedded ML solutions either rely on high-level runtimes with performance and memory overhead or focus primarily on deep learning models, resulting in complex deployment pipelines and limited applicability to resource-constrained devices. Moreover, current tools lack an automated and principled workflow that connects standard machine learning development in Python with efficient, native C/C++ deployment on Arduino-compatible platforms.

To address these challenges, this paper introduces **P-ML**, an end-to-end AutoML framework designed specifically for deploying classical machine learning models on memory-constrained embedded systems. P-ML automates the complete workflow, including data splitting, model selection, hyperparameter optimization, and generation of compact Arduino-compatible C++ libraries.

The framework integrates Optuna-based hyperparameter optimization with stratified data splitting strategies, including SPXY and K-Fold cross-validation, to ensure robust model evaluation. A wide range of classical machine learning algorithms is supported, and trained models are automatically converted into optimized C++ code suitable for Arduino Uno, Nano, and ESP32 platforms. The entire pipeline is unified within an easy-to-use interface, eliminating the need for manual configuration or specialized embedded systems expertise.

---

## 📜 License and Copyright Statement

The proposed framework and the generated Arduino/ESP32 libraries are released **exclusively for academic, research, and educational purposes**.

Users are allowed to use, modify, and reproduce the source code and generated libraries for **non-commercial use only**, provided that proper citation of this work is included in any related publications, technical reports, or derivative research.

Any form of **commercial use, redistribution for profit, sublicensing, or integration into proprietary or commercial systems is strictly prohibited** without prior written permission from the authors.

This work is archived and publicly available with a persistent identifier (DOI):

**DOI:**(https://doi.org/10.5281/zenodo.18130146)

**Copyright © 2026, Huu-Phuoc Nguyen**  
All rights reserved.

Third-party libraries or dependencies used within this framework remain subject to their respective licenses.

---

## 📚 Citation

If you use P-ML in your research or projects, please cite:

```bibtex
@article{nguyen2026pml,
  title={An End-to-End AutoML Framework for Deploying Classical Machine Learning Models on Resource-Constrained Devices},
  author={Nguyen, Huu-Phuoc},
  institution={Can Tho University},
  address={Can Tho, Vietnam},
  year={2026},
  doi={https://doi.org/10.5281/zenodo.18130146},
  note={Categories: B.4.2 (Embedded Systems), C.3.3 (Real-time and Embedded Systems), I.2.6 (Learning)}
}
```

---

<div align="center">

### 🌟 Star this repository if you find it helpful!

**Made with ❤️ for the Embedded Machine Learning Community**

[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--5822--4121-green.svg)](https://orcid.org/0009-0006-5822-4121)
[![Email](https://img.shields.io/badge/Email-huuphuoc.research%40gmail.com-red.svg)](mailto:huuphuoc.research@gmail.com)

---

**P-ML Framework** - Bringing Classical Machine Learning to the Edge  
© 2026 Huu-Phuoc Nguyen, Can Tho University

</div>









