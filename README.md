# Automatic Shifted Log Transformation
A Python module to automatically log transform your skewed data closer to normal distribution using method proposed by Feng et al (2016).
It is designed with TransformerMixin from Scikit-Learn so it can be used in Scikit-Learn Pipeline. The transformation is given by the following equation.
```math
\phi_\beta(X_i) = 
\begin{cases} 
\log\left(X_i - \min(X_1, X_2, \dots, X_n) + \left|\frac{1}{\beta}\right|R\right), & \beta > 0 \\ 
-\log\left(\max(X_1, X_2, \dots, X_n) - X_i + \left|\frac{1}{\beta}\right|R\right), & \beta < 0 
\end{cases}
```
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

# Example
Here is an example of ASLT with `beta=10`. You can tune `beta` manually by passing a list of the $$\beta$$ for each column to produce better results.
<br>
<img width="598" height="1004" alt="image" src="https://github.com/user-attachments/assets/3dea7fb0-f04f-467c-bfbd-4ac27c48d2d7" />
<br>
The data used to demonstrate this module is a cleaned TESS TOI Table from NASA Exoplanet Archive. The raw data is available at https://exoplanetarchive.ipac.caltech.edu/docs/data.html.

# Reference
Feng, Q., Hannig, J., & Marron, J. S. (2016). A note on automatic data transformation. Stat, 5(1), 82-87.
