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
    * Loaded gzipped JSON data into a Pandas DataFrame.
    * Filtered properties to `apartment` type within `Capital Federal` and `price_aprox_usd` < \$400,000.
    * Dropped columns with high percentages of missing values (`floor`, `expenses`).
    * Removed non-informative, constant, or highly cardinal metadata/text columns (`operation`, `property_type`, `currency`, `properati_url`, `title`, `description`, `image_thumbnail`, `geonames_id`, `created_on`).
    * Eliminated **leaky features** directly derived from the target (`price`, `price_aprox_local_currency`, `price_per_m2`, `price_usd_per_m2`).
    * Handled multicollinearity by dropping `surface_total_in_m2` and `rooms`.
    * Clipped outliers in `surface_covered_in_m2` using 1st and 99th percentiles (Winsorization).

2.  **Exploratory Data Analysis (EDA):**
    * Analyzed target variable distribution and class balance.
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

## Future Work

To further enhance the model's performance and robustness, the following steps are suggested:

* **Target Variable Transformation:** Apply a **log transformation** to the `price_aprox_usd` target variable to address heteroscedasticity and potentially improve predictions for higher-priced properties.
* **Outlier/Anomaly Handling:** Rigorously investigate and address the identified extreme outlier in `distance_to_center` (and potentially other features) to ensure data integrity.
* **Advanced Model Exploration:** Explore other powerful gradient boosting frameworks like **XGBoost** or **LightGBM**, which often outperform standard Gradient Boosting and Random Forests.
* **More Feature Engineering:** Investigate creating temporal features from `created_on` (e.g., listing age, month of listing), and explore more complex interaction terms between existing features.
* **Model Interpretability Deep Dive:** Utilize advanced interpretability libraries like SHAP or LIME to gain even deeper insights into individual predictions and feature contributions.
* **Model Deployment:** Consider building a simple API or web interface to deploy the model for practical price estimation.

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
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
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
Sawsan Yousef 
Data Scientist | Predictive Modeling | Computer Vision

[LinkedIn](https://www.linkedin.com/in/sawsan-yusuf-027b2b214?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) | [Medium](https://medium.com/@sawsanyusuf)


