# bayesian-neural-networks
Probabilistic Machine Learning project featuring Bayesian Linear Regression, MC Dropout, MCMC, and NUTS/HMC for robust uncertainty quantification in Neural Networks

# Bayesian Neural Networks (BNN) and Probabilistic Modeling

This repository contains the Phase 2 project for the **Introduction to Machine Learning (25737-2)** course at Sharif University of Technology (Spring 1404). 
The project bridges the gap between theoretical Bayesian foundations and practical implementation, focusing on uncertainty quantification in predictive modeling.

## 📌 Project Overview
Classical machine learning methods often provide deterministic point estimates, which lack the ability to quantify uncertainty. This project implements **Bayesian methodologies** to capture and quantify epistemic uncertainty, providing full posterior probability distributions for model parameters. This allows for the generation of credible intervals and confidence maps, essential for robust decision-making.

The project is implemented in Python within a comprehensive Jupyter Notebook (`IML_Project_Phase2.ipynb`) and covers four main modules:

### 1. Bayesian Linear Regression
*   **Dataset:** Portuguese high school student performance dataset (predicting final grade `G3`).
*   **Methodology:** Modeled using the `bambi` library (built on `PyMC`). Explored both Normal and Student-T prior distributions.
*   **Results:** The Student-T prior demonstrated greater robustness against outliers and heavy-tailed noise. The model successfully generated posterior distributions for parameters (e.g., negative weights for `failures` and `Dalc`, positive for `studytime` and `higher_yes`), allowing for uncertainty-aware predictions (e.g., 95% HDI intervals for new observations).

### 2. Bayesian Deep Learning via Dropout (Bonus)
*   **Methodology:** Leveraged **Monte Carlo (MC) Dropout** as a practical approximation to Bayesian inference in Neural Networks.
*   **Experiment 1 (1D Regression):** Modeled a noisy sinusoidal function. MC Dropout successfully captured predictive uncertainty, exhibiting wider confidence bands in regions with sparse training data.
*   **Experiment 2 (Image Classification):** Evaluated on the MNIST dataset with rotated images. The model effectively adjusted its confidence, showing higher predictive uncertainty for out-of-distribution rotation angles.

### 3. Basic Markov-Chain Monte Carlo (MCMC)
*   **Methodology:** Implemented the Metropolis-Hastings algorithm to sample from complex posterior distributions where analytical solutions are intractable.
*   **Results:** Successfully approximated posterior distributions for neural network parameters, allowing for the evaluation of model confidence alongside point predictions. 

### 4. Hamiltonian Monte Carlo (HMC) & NUTS
*   **Methodology:** Applied the No-U-Turn Sampler (NUTS), an advanced HMC algorithm that utilizes gradient information for highly efficient sampling in high-dimensional spaces.
*   **Experiment (Classification):** Modeled the non-linear `make_moons` dataset using a BNN.
*   **Results:** The BNN successfully learned a soft, probabilistic decision boundary. PCA trajectory plots of the HMC samples showed efficient exploration of the parameter space without getting stuck in local minima, converging to stable posterior regions.

## 🚀 How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed (e.g., `numpy`, `scipy`, `pytorch`, `pymc`, `bambi`, `arviz`).
3. Open the `notebooks/IML_Project_Phase2.ipynb` file and run the cells sequentially. The required datasets are located in the `data/` folder.

## 📂 Repository Structure
*   `/data`: Datasets used for regression and time-series modeling (`sunspot.dat`, `scaled_dataset.txt`, etc.).
*   `/notebooks`: The main Jupyter notebook containing the implementation of all 4 modules.
*   `/reports`: Detailed project reports with mathematical formulations, plots, and extensive analysis.
*   `/images`: Figures and trajectory plots generated during MCMC and HMC sampling.

## 👥 Authors
*   Mohammad Hossein Momeni
*   Amirhossein Naghdi
*   Hooman Zolfaghari
*   Ghazal Hosseini
*   Yahya Tehrani
*   Amir Afzali
*   Borna Khodabandeh
*   **Amir Ali Jafari (400100938)**
*   **Erfan Bateni (400100792)**
