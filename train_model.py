import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import r2_score

# ==============================================================================
# سلول ۱: تنظیمات اولیه
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("model_training")
ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
INPUT_CSV_PATH = os.path.join(DATA_DIR, "cleaned_sample.csv")
logger.info("Script started. Paths configured for your project structure.")

# ==============================================================================
# سلول ۲: بارگذاری و پیش‌پردازش با Feature Hashing
# ==============================================================================
try:
    logger.info("--- Loading and Preprocessing Data ---")
    df_clean = pd.read_csv(INPUT_CSV_PATH)
    df_clean.dropna(subset=['Price', 'Brand', 'Category1'], inplace=True)
    logger.info(f"Data loaded successfully. Shape: {df_clean.shape}")

    # --- جدا کردن بخش‌های مختلف داده ---
    numerical_cols = ['rate', 'likes', 'dislikes', 'is_buyer', 'product_id', 
                      'word_count', 'char_len', 'unique_word_ratio', 'emoji_count', 
                      'hour', 'dayofweek']
    categorical_cols = ['Brand', 'Category1']
    target_col = 'Price'

    df_numerical = df_clean[numerical_cols]
    df_categorical = df_clean[categorical_cols].astype(str) # تبدیل به رشته برای هشینگ
    y = np.log1p(df_clean[target_col])

    # ### <<< اصلاح کلیدی: استفاده از Feature Hashing >>> ###
    # به جای ساختن ۱۴۰۰ ستون، آن‌ها را در ۵۰ ستون فشرده می‌کنیم
    hasher = FeatureHasher(n_features=50, input_type='string')
    hashed_features = hasher.fit_transform(df_categorical.values)
    
    # تبدیل نتیجه به یک دیتافریم با نام‌های ستون مشخص
    hashed_df = pd.DataFrame(hashed_features.toarray(), 
                             columns=[f'cat_hash_{i}' for i in range(50)])
    logger.info(f"Applied Feature Hashing. Shape of hashed features: {hashed_df.shape}")

    # --- ترکیب ویژگی‌های عددی و هش‌شده ---
    X = pd.concat([df_numerical.reset_index(drop=True), hashed_df.reset_index(drop=True)], axis=1)
    logger.info(f"Combined numerical and hashed features. Final X shape: {X.shape}")

    # --- ذخیره‌سازی لیست ستون‌ها و مقادیر پیش‌فرض ---
    with open(os.path.join(MODELS_DIR, "model_columns.json"), 'w', encoding='utf-8') as f:
        json.dump(X.columns.tolist(), f, ensure_ascii=False, indent=4)
    logger.info(f"SUCCESS: Saved 'model_columns.json' with {len(X.columns)} columns.")
    
    default_values = df_numerical.mean().to_dict()
    with open(os.path.join(MODELS_DIR, "default_values.json"), 'w', encoding='utf-8') as f:
        json.dump(default_values, f, ensure_ascii=False, indent=4)
    logger.info("SUCCESS: Saved 'default_values.json'.")

except Exception as e:
    logger.error(f"An error occurred during preprocessing: {e}", exc_info=True)
    raise

# ==============================================================================
# سلول ۳: آموزش و ذخیره‌سازی مدل (بدون تغییر منطقی)
# ==============================================================================
try:
    logger.info("--- Splitting, Scaling, and Training ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20) # افزایش عمق برای یادگیری بهتر
    
    logger.info("Training RandomForestRegressor model on HASHED features...")
    rf_model.fit(X_train_scaled, y_train)
    logger.info("Model training complete.")

    # --- ذخیره‌سازی نهایی ---
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "your_model.pkl"))
    logger.info("SUCCESS: Saved final model to 'models/your_model.pkl'.")
    joblib.dump(scaler, os.path.join(MODELS_DIR, "regression_scaler.pkl"))
    logger.info("SUCCESS: Saved final scaler to 'models/regression_scaler.pkl'.")
    # ذخیره هشر برای استفاده در داشبورد
    joblib.dump(hasher, os.path.join(MODELS_DIR, "feature_hasher.pkl"))
    logger.info("SUCCESS: Saved feature hasher to 'models/feature_hasher.pkl'.")

    # --- ارزیابی نهایی ---
    y_pred_log = rf_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_log)
    logger.info(f"Final Model R^2 Score: {r2:.4f}")
    logger.info("--- Script Finished Successfully! ---")

except Exception as e:
    logger.error(f"An error occurred during training/saving: {e}", exc_info=True)
    raise

