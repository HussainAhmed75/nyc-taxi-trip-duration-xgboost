# NYC Taxi Trip Duration Prediction

Predicting how long a New York taxi trip will take from the pickup time and the distance between pickup and drop-off, using XGBoost.

## Data

[NYC Taxi Trip Duration](https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration) (Kaggle): **1,458,644 trips** from 2016 with pickup/drop-off times, GPS coordinates, passenger count and vendor. No missing values and no duplicates.

## Approach

**Feature engineering**
- Time features from `pickup_datetime`: hour, day, month, weekday, weekend flag
- `distance_km`: Haversine distance between pickup and drop-off coordinates
- `speed_kmh` (used only for cleaning)

**Cleaning**
- Removed outliers in trip duration and distance with the IQR rule
- Removed physically unrealistic trips (speed below 1 km/h or above 80 km/h)
- About **1.27M trips** remain after cleaning

**Target**
- Trip duration is heavily right-skewed, so the model predicts `log1p(trip_duration)`

**Model**
- Features: `passenger_count`, `pickup_hour`, `pickup_day`, `pickup_weekday`, `pickup_is_weekend`, `distance_km`
- MinMax scaling, 80/20 train/test split
- `XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, subsample=0.8, colsample_bytree=0.8)`

## Results (test set, log scale)

| Metric | Train | Test |
|---|---|---|
| R² | 0.647 | 0.641 |
| MAE | 0.302 | 0.304 |
| RMSE | 0.396 | 0.399 |

Train and test scores are almost identical, so the model is not overfitting. An MAE of about 0.30 on the log scale means predictions are typically within roughly ±35% of the real trip time.

## Demo

`app.py` includes a small Gradio app: enter passengers, pickup time and distance, and it returns the predicted trip duration.

```bash
git clone https://github.com/HussainAhmed75/nyc-taxi-trip-duration-xgboost.git
cd nyc-taxi-trip-duration-xgboost
pip install -r requirements.txt
```

Then open `Notebook.ipynb` (the full EDA, cleaning and training workflow) or run `app.py`. The data is downloaded automatically with `kagglehub`.

## What I'd improve next

- Fix the weekend flag (it currently marks Friday only instead of Saturday/Sunday)
- Add location features (pickup/drop-off zones or clusters), since distance alone misses traffic patterns between areas
- Compare with LightGBM/CatBoost and tune hyperparameters
- Report errors in minutes, not only on the log scale

## Author

Hussain Ahmed · [LinkedIn](https://www.linkedin.com/in/hussian-ahmed) · hrfy330@gmail.com
