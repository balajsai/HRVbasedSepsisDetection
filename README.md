Harnessing Heart Rate Variability for Sepsis Detection
Overview

This project is based on the original research, "Harnessing Heart Rate Variability: A Machine Learning Approach to the Early Detection and Predictive Monitoring of Sepsis3," by Sai Balaji at the Academy of Science and Technology, The Woodlands College Park High School, The Woodlands, Texas, USA. The study focuses on developing an effective predictive model for early Sepsis3 detection by leveraging Heart Rate Variability (HRV) as a biomarker and utilizing various machine learning techniques, including XGBoost, Random Forest, and neural networks.
Abstract

The study highlights the application of specific feature selection methods to enhance the model's performance, which yielded significant metrics of precision, recall, and F1-score. Furthermore, the interpretability of the model was augmented through Local Interpretable Model-agnostic Explanations (LIME), demonstrating its potential as a reliable sepsis alert system. Through this research, the efficacy of HRV in predicting sepsis is underscored, providing a foundation for further advancements in predictive medical monitoring systems.
Requirements

To run the scripts in this project, you will need Python installed along with several data science libraries. Please refer to requirements.txt for a detailed list of dependencies.
Setup Instructions
Clone this repository to your local machine:

    git clone [HRVbasedSepsisDetection](https://github.com/balajsai/HRVbasedSepsisDetection.git)

Install the required libraries:

    pip install -r requirements.txt

Run the Python script:

    python hrvsepsis3.py


The dataset used in this research, authored by Kuan-Fu Chen from Chang Gung University College of Medicine, is titled “Derivation and Validation of Heart Rate Variability-Based Scoring Models for Continuous Risk Stratification in Patients with Suspected Sepsis in the Emergency Department,” published on 9 December 2020\cite{Chen}. This HRV sepsis dataset provides a comprehensive collection of heart rate variability metrics, essential for developing machine learning models to detect early signs of Sepsis3, thereby offering a multidimensional representation of HRV as potential biomarkers for diagnostics in the clinical context.
