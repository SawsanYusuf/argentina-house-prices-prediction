# Apartment Price Prediction in Buenos Aires, Capital Federal

## Project Overview

This project develops a machine learning model to accurately predict apartment prices in the Capital Federal autonomous city of Buenos Aires, Argentina. Leveraging a comprehensive dataset of property listings, the goal is to provide reliable price estimations for various stakeholders in the real estate market. The workflow encompasses data cleaning, extensive feature engineering, robust model training, and thorough evaluation to ensure a high-performing and interpretable predictive solution.

## Key Features

* **Data Wrangling:** Efficient loading and cleaning of raw property data, including handling missing values and initial data subsetting.
* **Exploratory Data Analysis (EDA):** In-depth analysis of feature distributions, correlations, and spatial relationships to understand price drivers.
* **Advanced Feature Engineering:** Creation of impactful spatial features such as `distance_to_center`, `neighborhood_avg_price` (target encoding), and `lat_lon_cluster` (geographical clustering) to enhance predictive power.
* **Robust Preprocessing Pipeline:** Implementation of `ColumnTransformer` and `Pipeline` for consistent imputation, scaling, and encoding of features, preventing data leakage.
* **Model Training & Hyperparameter Tuning:** Systematic training and optimization of various regression models, with a focus on Random Forest Regressor using `GridSearchCV`.
* **Comprehensive Evaluation:** Rigorous assessment of model performance using Mean Absolute Error (MAE) and R-squared (R²) on an unseen test set.
* **Model Interpretability:** Analysis of feature importances to understand key price drivers and residual analysis to diagnose model errors.

## Dataset

The project utilizes a historical dataset of property listings from **Properati**, specifically focusing on apartment sales in **Capital Federal, Buenos Aires, Argentina**, with prices under \$400,000 USD. The dataset includes various property attributes such as:

* `lat`, `lon` (latitude and longitude)
* `place_name` (neighborhood)
* `surface_covered_in_m2` (covered area in square meters)
* `price_aprox_usd` (approximate price in USD - **Target Variable**)
* And other metadata features handled during wrangling.

## Methodology

The project followed a structured machine learning pipeline:

1.  **Data Loading & Initial Filtering (`wrangle` function):**
    * Loaded CSV data into a Pandas DataFrame.
    * Filtered properties to `apartment` type within `Capital Federal` and `price_aprox_usd` < \$400,000.
    * Dropped columns with high percentages of missing values (`floor`, `expenses`).
    * Removed non-informative, constant, or highly cardinal metadata/text columns (`operation`, `property_type`, `currency`, `properati_url`, `title`, `description`, `image_thumbnail`, `geonames_id`, `created_on`).
    * Eliminated **leaky features** directly derived from the target (`price`, `price_aprox_local_currency`, `price_per_m2`, `price_usd_per_m2`).
    * Handled multicollinearity by dropping `surface_total_in_m2` and `rooms`.
    * Clipped outliers in `surface_covered_in_m2` using 1st and 99th percentiles (Winsorization).

2.  **Exploratory Data Analysis (EDA):**
    * Analyzed target variable distribution.
    * Visualized distributions of numerical features (histograms) and their relationships with the target (box plots).
    * Generated correlation heatmaps to understand inter-feature relationships.
    * Explored spatial patterns of price using interactive plots (though not explicitly shown in final code, it was part of the process).

3.  **Feature Engineering:**
    * `distance_to_center`: Calculated the geodesic distance from each property's coordinates to a central point in Capital Federal.
    * `neighborhood_avg_price`: Created a target-encoded feature by mapping each `place_name` to the average `price_aprox_usd` of properties in that neighborhood (fitted on training data only).
    * `lat_lon_cluster`: Applied K-Means clustering to `lat` and `lon` to create a new categorical feature representing geographical zones (fitted on training data only).

4.  **Data Splitting & Preprocessing Pipeline:**
    * Split the dataset into `X_train`, `X_val`, `X_test` (and corresponding `y` sets) to ensure robust evaluation.
    * Built a `ColumnTransformer`-based preprocessing pipeline for:
        * Numerical features: `SimpleImputer` (mean strategy) and `StandardScaler`.
        * Categorical features: `SimpleImputer` (constant strategy) and `OneHotEncoder` (`handle_unknown='ignore'`).

5.  **Model Training & Hyperparameter Tuning:**
    * Utilized a custom `run_model` function to train and evaluate various regression models (Linear Regression, Ridge, Lasso, Random Forest Regressor, Gradient Boosting Regressor).
    * Performed `GridSearchCV` for Random Forest Regressor, systematically searching for optimal hyperparameters (e.g., `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`) to minimize MAE on cross-validation folds.

6.  **Model Evaluation & Diagnostics:**
    * Evaluated the final, best-performing Random Forest model (with spatial features) on the completely unseen `X_test` set.
    * Conducted **Feature Importance** analysis to identify key price drivers.
    * Performed **Residual Analysis** (histogram of residuals, residuals vs. predicted values plots) to diagnose model errors and identify areas for improvement.

## Results

The final Random Forest Regressor model, leveraging engineered spatial features, achieved the following performance on the unseen test set:

* **Mean Absolute Error (MAE): 19268.38 USD**
* **R-squared (R²): 0.7559**

This indicates that, on average, the model's predictions are within approximately \$19,268 USD of the actual apartment price, and it explains about 75.6% of the variance in apartment prices.

**Key Findings from Feature Importance:**
The most influential features driving apartment prices were:
1.  `surface_covered_in_m2`
2.  `neighborhood_avg_price` (engineered feature)
3.  `lat`
4.  `lon`
5.  `distance_to_center` (engineered feature)

**Residual Analysis Observations:**
The residuals were largely centered around zero, suggesting no systematic bias. However, the analysis revealed **heteroscedasticity** (a "fan shape"), indicating larger prediction errors for higher-priced properties. An extreme outlier in the `distance_to_center` feature was also identified as a potential data anomaly.


## Post-Project Revisit: Empirical Findings & Lessons Learned 

This section documents a comprehensive post-project revisit conducted approximately one year later. The goal was to empirically test earlier assumptions, evaluate proposed future work, and determine the true performance limits of the dataset.


### 1. Establishing the Baseline: Raw Data Reality

The raw dataset exhibited severe statistical distortions caused by unrealistic outliers, such as apartments with zero surface area and properties priced far beyond plausible market ranges.

Initial linear models failed almost entirely, as they attempted to accommodate extreme points far removed from the underlying data distribution.

**Turning Point:**
Applying **clipping (removal of the top and bottom 10%)** to key numerical variables marked the most impactful intervention in the entire project. Without any additional modeling changes, linear model performance improved dramatically:

* Before clipping: R² ≈ 0.23

* After clipping: R² ≈ 0.65

**Key Insight:**
Data cleaning alone had a greater impact on model performance than any subsequent transformation or algorithmic refinement.


### 2. The Log Transformation Hypothesis: Tested, Not Assumed

To address the heteroscedasticity observed in residual plots, log-based transformations were systematically evaluated.

#### Log Transformation on Target Only

* Did not improve linear model performance and introduced instability.
* Transforming price while keeping surface area linear shifted the model toward percentage-based reasoning, creating a mismatch in feature–target scale logic.

#### Log Transformation on Target and Features

* Severe degradation of linear models.
* Lasso collapsed entirely, yielding negative R² values.
* After transformation, surface area values became very small while latitude and longitude remained large-scale features, causing the models to effectively ignore surface area.

#### Selective Log Transformation with Scaling

* Log applied only to price and surface area, excluding spatial coordinates, combined with `StandardScaler`.
* Models stabilized but failed to surpass an R² ceiling of approximately **0.61**.

**Conclusion:**
Log transformation is not a universal remedy. In this dataset, it increased relational complexity rather than reducing it, particularly in the presence of dominant spatial effects.


### 3. Algorithmic Behavior: Linear vs. Tree-Based Models

#### Linear Models (Linear / Ridge / Lasso)

* Performance plateaued around **R² ≈ 0.65** after clipping.
* The relationship between geography and price is inherently non-linear; price changes occur in discrete jumps between neighborhoods rather than smooth gradients.

#### Tree-Based Models (Random Forest & Gradient Boosting)

* **Random Forest** consistently achieved **R² ≈ 0.75–0.77** and emerged as the strongest model.
* It effectively captured geographic boundaries and price clustering without requiring complex transformations.


### 4. Spatial Noise & Model Robustness

**Empirical Evidence of Noise:**

* Geospatial visualizations revealed significant price dispersion even among properties at nearly identical coordinates.
* Neighborhood-level box plots showed wide interquartile ranges and heavy overlap, confirming strong intra-neighborhood variability.

This indicates that a large portion of price variance originates from unobserved factors such as building condition, amenities, or views.

**Model Implications:**

* **Random Forest** demonstrated higher robustness due to bagging and variance reduction.
* **Gradient Boosting**, while theoretically powerful, proved more sensitive to noise and prone to overfitting under a limited feature space.


### 5. Gradient Boosting: Why It Did Not Outperform

Despite common theoretical expectations, Gradient Boosting consistently underperformed Random Forest in this project.

* The dataset contains high noise and only **four effective features** after preprocessing.
* Gradient Boosting’s sequential error-correction mechanism amplified noise when the signal-to-noise ratio was low.
* Random Forest provided a better bias–variance trade-off and generalized more effectively.


### 6. Generalization & Stability Check

Validation and test prediction plots showed near-identical scatter patterns and error distributions.

**Implications:**

1. No evidence of overfitting.
2. Stable generalization to unseen data.
3. Model readiness for practical use on future data from the same city.


### 7. Performance Ceiling & Final Takeaways

With only **four meaningful features**, achieving **~77% explained variance** represents a practical upper bound for this dataset.

Further model tuning or increased algorithmic complexity is unlikely to yield meaningful gains. Real improvement requires **new information**, not new equations.

**Core Lessons:**

* Data quality outweighs model sophistication.
* Log transformation is contextual, not sacred.
* Knowing when to stop is a critical applied ML skill.


## Future Work 

Given the observed performance ceiling, future improvements are expected to come primarily from **data enrichment rather than model complexity**:

* **Feature Expansion:** Incorporate additional informative features such as:

  * Building age
  * Number of bathrooms
  * Floor level and total floors
  * Property amenities (elevator, parking, balcony, sea view)
* **External Data Sources:** Integrate neighborhood-level socio-economic indicators or accessibility metrics.
* **Temporal Signals:** Introduce time-based features (market trends, seasonality).
* **Model Deployment:** Build a lightweight API or web interface for real-world price estimation.

Further experimentation with more complex models is unlikely to outperform Random Forest without expanding the feature space.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* geopy (for geodesic distance calculations)
* `category_encoders` (for OneHotEncoder, could be used for TargetEncoder)

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SawsanYusuf/argentina-house-prices-prediction.git](https://github.com/SawsanYusuf/argentina-house-prices-prediction.git)
    cd argentina-house-prices-prediction
    ```
2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn geopy category_encoders
    ```
3.  **Download the dataset:**
    Ensure the `properati-AR-2016-11-01-properties-sell.csv.gz` (or `.csv` if not gzipped) file is placed in the correct path as referenced in the `wrangle` function (e.g., `/content/properati-AR-2016-11-01-properties-sell.csv`).
4.  **Run the Jupyter Notebook/Colab:**
    Open the `apartment_price_prediction.ipynb` (or your Colab notebook) and run all cells sequentially.

## Author
**Sawsan Yousef** 

*Data Scientist | Predictive Modeling | Computer Vision*

[LinkedIn](https://www.linkedin.com/in/sawsan-yusuf-027b2b214?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) | [Medium](https://medium.com/@sawsanyusuf)


