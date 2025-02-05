# ConformaSegment Time-Series Explainer Package 📦

![Static Badge](https://img.shields.io/badge/concept-XAI-lightblue?style=flat)
![Static Badge](https://img.shields.io/badge/concept-uncertainty-blue?style=flat&logoColor=white&color=C46210)
![Static Badge](https://img.shields.io/badge/python-package-gold?style=flat&logo=python&logoColor=white&color=blue)
![Static Badge](https://img.shields.io/badge/library-scikit_learn-blue?style=flat&logo=scikit-learn&logoColor=white&color=C46210)
![Static Badge](https://img.shields.io/badge/library-tensorflow-blue?style=flat&logo=tensorflow&logoColor=white&color=C46210)
![Static Badge](https://img.shields.io/badge/jupyter-notebooks-orange?style=flat&logo=jupyter&logoColor=white&color=orange)
![Static Badge](https://img.shields.io/badge/version-v1.0-green?style=flat&logo=github&logoColor=white&color=5533FF)
![Static Badge](https://img.shields.io/badge/documentation-paper-blue?style=flat&logo=github&logoColor=white&color=660055)

### Introduction

ConformaSegment is an innovative framework built on conformal prediction, designed to generate robust and interpretable interval-based explanations for time-series forecasting—without relying on specific data distributions. 
By identifying time intervals that significantly influence whether actual values fall within predicted confidence bounds, the framework sheds light on the model’s decision-making process. The presence of such impactful intervals indicates their essential role in shaping predictive confidence. 
Thus, effective explainability demands not only pinpointing these key periods but also understanding how fluctuations in uncertainty alter forecasting outcomes. 
ConformaSegment addresses these challenges by emphasizing the most crucial intervals, providing insights that enhance both reliability and interpretability. This foundation gives rise to critical research questions that drive further exploration into interval-based explanations in forecasting models.

```python
sample = X_test[143]
label_sample = y_test[143]

error_rate = 0.10
pelt_penalty_lambda = 2

cs.get_feature_importance(regressor, sample, label_sample, X_cal, y_cal, error_rate, pelt_penalty_lambda)
```


