# import pandas as pd
# import numpy as np
# import joblib
# from tensorflow.keras.models import load_model
# import json
# import os
# import re
# import logging
# from sklearn.preprocessing import StandardScaler

# # --- تنظیمات اولیه ---
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
# logger = logging.getLogger("shared_data_loader")

# OUTPUT_DIR = "."
# MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
# DATA_DIR = os.path.join(OUTPUT_DIR, "data")

# # --- توابع پیش‌پردازش متن (مشترک) ---
# def simple_normalize(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.replace('ي', 'ی').replace('ك', 'ک')
#     return text

# def clean_text(text: str) -> str:
#     if not isinstance(text, str): return ""
#     text = simple_normalize(text)
#     text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
#     text = re.sub(r"[^آ-یA-Za-z0-9\s]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def emoji_count(text: str) -> int:
#     if not isinstance(text, str): return 0
#     return len(re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE).findall(text))

# # --- بارگذاری مدل‌ها و داده‌ها ---
# def load_resource(loader, path, description):
#     try:
#         resource = loader(path)
#         logger.info(f"{description} loaded successfully from {path}")
#         return resource
#     except Exception as e:
#         logger.error(f"Failed to load {description} from {path}: {e}")
#         return None

# rf_model = load_resource(joblib.load, os.path.join(MODELS_DIR, "your_model.pkl"), "RandomForest Regressor")
# regression_scaler = load_resource(joblib.load, os.path.join(MODELS_DIR, "regression_scaler.pkl"), "Regression Scaler")
# try:
#     with open(os.path.join(MODELS_DIR, "model_columns.json"), 'r', encoding='utf-8') as f:
#         model_columns = json.load(f)
#     with open(os.path.join(MODELS_DIR, "default_values.json"), 'r', encoding='utf-8') as f:
#         default_values = json.load(f)
# except Exception as e:
#     logger.error(f"Failed to load JSON files: {e}")
#     model_columns = []
#     default_values = {}
# best_classifier = load_resource(joblib.load, os.path.join(MODELS_DIR, "best_classifier_model.pkl"), "Best Classifier")
# label_encoder = load_resource(joblib.load, os.path.join(MODELS_DIR, "label_encoder.pkl"), "Label Encoder")
# autoencoder = load_resource(lambda p: load_model(p, compile=False), os.path.join(MODELS_DIR, "autoencoder_model.h5"), "Autoencoder")
# tfidf = load_resource(joblib.load, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "TF-IDF Vectorizer")
# svd = load_resource(joblib.load, os.path.join(MODELS_DIR, "svd_model.pkl"), "SVD Model")
# feature_scaler = load_resource(joblib.load, os.path.join(MODELS_DIR, "feature_scaler.pkl"), "Feature Scaler")
# df_preprocessed = load_resource(pd.read_csv, os.path.join(DATA_DIR, "cleaned_sample.csv"), "Preprocessed DataFrame")
# X_scaled = load_resource(np.load, os.path.join(DATA_DIR, "X_scaled.npy"), "Scaled Features (X_scaled)")
# X_pca = load_resource(np.load, os.path.join(DATA_DIR, "X_pca.npy"), "Pre-calculated PCA Features")
# df_out = load_resource(pd.read_csv, os.path.join(DATA_DIR, "df_out_with_scores.csv"), "Anomaly Scores DataFrame")
# comparison_df = load_resource(lambda p: pd.read_csv(p, index_col=0), os.path.join(DATA_DIR, "model_comparison.csv"), "Model Comparison DataFrame")
# brands = df_preprocessed['Brand'].unique().tolist() if df_preprocessed is not None and 'Brand' in df_preprocessed.columns else ['unknown']
# categories = df_preprocessed['Category1'].unique().tolist() if df_preprocessed is not None and 'Category1' in df_preprocessed.columns else ['unknown']

# # --- CHANGE: Rewritten function for robust one-hot encoding ---
# def preprocess_for_regression(brand, category1):
#     """
#     Prepares a single row DataFrame for prediction.
#     It creates a zero-vector and sets the appropriate one-hot encoded columns to 1.
#     """
#     if rf_model is None or not model_columns:
#         logger.error("Regression model or model_columns not loaded.")
#         return None
    
#     try:
#         # ۱. یک دیتافریم با یک ردیف صفر و ستون‌های هماهنگ با مدل بساز
#         final_input = pd.DataFrame(0, index=[0], columns=model_columns)

#         # ۲. مقادیر پیش‌فرض را برای ستون‌های عددی تنظیم کن
#         for col, value in default_values.items():
#             if col in final_input.columns:
#                 final_input.at[0, col] = value
        
#         # ۳. ستون مربوط به برند انتخاب شده را پیدا و مقدار آن را ۱ کن
#         brand_col = f"Brand_{brand}"
#         if brand_col in final_input.columns:
#             final_input.at[0, brand_col] = 1
        
#         # ۴. ستون مربوط به دسته‌بندی انتخاب شده را پیدا و مقدار آن را ۱ کن
#         category_col = f"Category1_{category1}"
#         if category_col in final_input.columns:
#             final_input.at[0, category_col] = 1
            
#         return final_input

#     except Exception as e:
#         logger.error(f"Error in preprocess_for_regression: {e}", exc_info=True)
#         return None

# def preprocess_for_anomaly(body, rate, likes, dislikes, price):
#     # ... (این تابع بدون تغییر باقی می‌ماند)
#     if tfidf is None or svd is None or feature_scaler is None: return None
#     try:
#         clean_body = clean_text(body)
#         word_count = len(clean_body.split())
#         char_len = len(clean_body)
#         unique_word_ratio = len(set(clean_body.split())) / max(1, word_count)
#         emoji_cnt = emoji_count(body)
#         tfidf_vec = tfidf.transform([clean_body])
#         tfidf_dense = svd.transform(tfidf_vec)
#         numeric_features = np.array([[
#             char_len, word_count, unique_word_ratio, emoji_cnt,
#             rate if rate is not None else 3,
#             likes if likes is not None else 0,
#             dislikes if dislikes is not None else 0,
#             price if price is not None else 1000000
#         ]])
#         numeric_scaled = feature_scaler.transform(numeric_features)
#         features = np.hstack([tfidf_dense, numeric_scaled])
#         return StandardScaler().fit_transform(features)
#     except Exception as e:
#         logger.error(f"Error in preprocess_for_anomaly: {e}")
#         return None


import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from tensorflow.keras.models import load_model # برای مدل Autoencoder

# --- تنظیمات اولیه و لاگ ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("shared_data_loader")

# --- تعریف مسیرها ---
ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# --- تابع کمکی برای بارگذاری فایل‌ها ---
def load_resource(loader, path, description):
    try:
        resource = loader(path)
        logger.info(f"{description} loaded successfully from {path}")
        return resource
    except Exception as e:
        logger.error(f"Failed to load {description} from {path}: {e}", exc_info=False) # exc_info=False برای تمیز بودن لاگ
        return None

# --- بارگذاری داده‌ها و مدل‌ها ---
logger.info("--- Loading all assets for the dashboard ---")
df_preprocessed = load_resource(pd.read_csv, os.path.join(DATA_DIR, "cleaned_sample.csv"), "Preprocessed DataFrame")
brands = sorted(df_preprocessed['Brand'].dropna().unique().tolist()) if df_preprocessed is not None and 'Brand' in df_preprocessed.columns else ['...']
categories = sorted(df_preprocessed['Category1'].dropna().unique().tolist()) if df_preprocessed is not None and 'Category1' in df_preprocessed.columns else ['...']

# --- بخش ۱: بارگذاری مدل‌های رگرسیون ---
rf_model = load_resource(joblib.load, os.path.join(MODELS_DIR, "your_model.pkl"), "RandomForest Regressor")
regression_scaler = load_resource(joblib.load, os.path.join(MODELS_DIR, "regression_scaler.pkl"), "Regression Scaler")
feature_hasher = load_resource(joblib.load, os.path.join(MODELS_DIR, "feature_hasher.pkl"), "Feature Hasher")
model_columns = load_resource(lambda p: json.load(open(p, 'r', encoding='utf-8')), os.path.join(MODELS_DIR, "model_columns.json"), "Model Columns") or []
default_values = load_resource(lambda p: json.load(open(p, 'r', encoding='utf-8')), os.path.join(MODELS_DIR, "default_values.json"), "Default Values") or {}

# --- بخش ۲: بارگذاری مدل‌های صفحات دیگر (طبقه‌بندی و ناهنجاری) ---
best_classifier = load_resource(joblib.load, os.path.join(MODELS_DIR, "best_classifier_model.pkl"), "Best Classifier")
label_encoder = load_resource(joblib.load, os.path.join(MODELS_DIR, "label_encoder.pkl"), "Label Encoder")
autoencoder = load_resource(lambda p: load_model(p, compile=False), os.path.join(MODELS_DIR, "autoencoder_model.h5"), "Autoencoder")
tfidf = load_resource(joblib.load, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "TF-IDF Vectorizer")
svd = load_resource(joblib.load, os.path.join(MODELS_DIR, "svd_model.pkl"), "SVD Model")
feature_scaler = load_resource(joblib.load, os.path.join(MODELS_DIR, "feature_scaler.pkl"), "Feature Scaler")
X_scaled = load_resource(np.load, os.path.join(DATA_DIR, "X_scaled.npy"), "Scaled Features (X_scaled)")
X_pca = load_resource(np.load, os.path.join(DATA_DIR, "X_pca.npy"), "Pre-calculated PCA Features")
df_out = load_resource(pd.read_csv, os.path.join(DATA_DIR, "df_out_with_scores.csv"), "Anomaly Scores DataFrame")
comparison_df = load_resource(lambda p: pd.read_csv(p, index_col=0), os.path.join(DATA_DIR, "model_comparison.csv"), "Model Comparison DataFrame")
logger.info("--- All assets loaded ---")


# --- تابع پیش‌پردازش برای رگرسیون ---
def preprocess_for_regression(brand, category1):
    if not all([rf_model, regression_scaler, model_columns, default_values, feature_hasher]):
        return None
    try:
        numerical_df = pd.DataFrame([default_values])
        categorical_data = [[str(brand), str(category1)]]
        hashed_features = feature_hasher.transform(categorical_data)
        hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f'cat_hash_{i}' for i in range(feature_hasher.n_features)])
        combined_df = pd.concat([numerical_df, hashed_df], axis=1)
        final_df_ordered = combined_df[model_columns]
        scaled_data = regression_scaler.transform(final_df_ordered)
        final_input = pd.DataFrame(scaled_data, columns=model_columns)
        return final_input
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        return None

