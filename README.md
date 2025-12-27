# Physics-Informed-ML: Soil Moisture Prediction with PINNs

## 📌 Context & Overview
Traditional Deep Learning models often act as "black boxes," ignoring the fundamental laws of physics. In eco-hydrology, soil moisture dynamics follow mass conservation principles. This project explores the implementation of **Physics-Informed Neural Networks (PINNs)** to predict the Normalized Difference Water Index (NDWI) while integrating water mass balance constraints into the loss function.

## 🎯 Objectives
* **Hybrid Modeling:** Combining data-driven learning (satellite indices) with physical constraints (water conservation).
* **PINN Implementation:** Developing a custom loss function that penalizes predictions violating hydrological laws.
* **Experimental Benchmarking:** Assessing the stability of neural networks when forced to respect physical boundaries.

## 🛠️ Tech Stack & Methodology
* **Language/Platform:** Python (Google Colab).
* **Deep Learning:** `TensorFlow` or `PyTorch` (Fully Connected Architecture).
* **Physics Integration:** Custom Loss Function defined as:
  $$Loss_{total} = MSE_{data} + \lambda_{phys} \cdot MSE_{physics}$$
* **Parameters:** Input features include LST (Temperature), NDVI (Vegetation), and Precipitation for 313 sub-basins.



## 🚀 Experimental Insights & Results
* **The PINN Challenge:** The model faced stability issues, highlighting the difficulty of balancing data accuracy with physical consistency in complex environmental systems.
* **Model Diagnostics:** Predictions showed an underestimation of NDWI, likely due to a simplified formalization of the water balance equation ($\Delta W \approx P - ET$).
* **Critical Learning:** The project demonstrated that PINNs require highly calibrated datasets and precise mathematical definitions of physical residuals to converge effectively.
* **Methodological Value:** Despite the convergence hurdles, the experiment provides a foundational framework for integrating Earth Observation data with differential equations in Python.

## 🔮 Perspectives for Improvement
* **Refined Formalization:** Improving the Evapotranspiration (ET) estimation using LST and NDVI through Penman-Monteith derived proxies within the network.
* **Architectural Search:** Testing Recurrent Neural Networks (LSTM-PINN) to better capture the temporal memory of soil moisture.
* **Data Normalization:** Implementing more robust scaling techniques to align disparate satellite and climatic variables.
* **Dynamic Penalization:** Using self-adaptive weights ($\lambda_{phys}$) to balance the loss terms during training epochs.

Link https://colab.research.google.com/drive/1XuZI-yg0GKGwTNFyXmstVwE0xU9Co3UA
