# Automatic Shifted Log Transformation
A Python module to automatically log transform your skewed data closer to normal distribution using method proposed by Feng et al (2016).
It is designed with TransformerMixin from Scikit-Learn so it can be used in Scikit-Learn Pipeline.

# Usage
The ASLT class can be used directly or through Scikit-Learn estimator. Here are some example on how to use this module.
### Direct Application
```
from autolog import ASLT

aslt = ASLT(beta=10)
X_transformed_train = aslt.fit_transform(X_train)
X_transformed_test = aslt.transform(X_test)
```
### Through Scikit-Learn Estimator
```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from autolog import ASLT

pl = Pipeline([
    ('imputer',KNNImputer()),
    ('aslt',ASLT(beta=10)),
    ('scaler',StandardScaler())
])
X_transformed_train = pl.fit_transform(X_train)
X_transformed_test = pl.transform(X_test)
```

# Note
- Make sure your data does not have any nan values, as it appears to break the ASLT module.
- You can run a nan imputer first such as `sklearn.impute.KNNImputer` before ASLT.

# Reference
Feng, Q., Hannig, J., & Marron, J. S. (2016). A note on automatic data transformation. Stat, 5(1), 82-87.
