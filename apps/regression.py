# from dash import dcc, html, callback, Input, Output, State
# import dash_bootstrap_components as dbc
# import numpy as np

# from app import app
# import shared_data  # وارد کردن مدل‌ها و داده‌های بارگذاری شده

# # --- لایه ظاهری صفحه ---
# layout = dbc.Container([
#     dbc.Row([
#         dbc.Col(html.H2("پیش‌بینی قیمت محصول", className="text-center"), width=12)
#     ]),

#     dbc.Row([
#         # ستون ورودی‌ها
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("ویژگی‌های محصول را انتخاب کنید"),
#                 dbc.CardBody([
#                     dbc.Form([
#                         dbc.Label("برند:", html_for="reg-brand", className="mt-3"),
#                         dcc.Dropdown(
#                             id='reg-brand',
#                             options=[{'label': b, 'value': b} for b in shared_data.brands],
#                             value=None,  # مقدار اولیه خالی است
#                             placeholder="یک برند را انتخاب کنید...",
#                             searchable=True
#                         ),

#                         dbc.Label("دسته‌بندی:", html_for="reg-category", className="mt-3"),
#                         dcc.Dropdown(
#                             id='reg-category',
#                             options=[{'label': c, 'value': c} for c in shared_data.categories],
#                             value=None,  # مقدار اولیه خالی است
#                             placeholder="یک دسته‌بندی را انتخاب کنید...",
#                             searchable=True
#                         ),

#                         dbc.Button("پیش‌بینی قیمت", id="predict-price-btn", color="primary", className="mt-4 w-100"),
#                     ])
#                 ])
#             ])
#         ], md=6),

#         # ستون خروجی
#         dbc.Col([
#             dbc.Card([
#                 dbc.CardHeader("نتیجه پیش‌بینی"),
#                 dbc.CardBody(
#                     html.Div(id='price-prediction-output', children=[
#                         html.H4("قیمت پیش‌بینی شده:", className="text-center"),
#                         html.H2("...", id="predicted-price-text", className="text-center text-success fw-bold"),
#                         html.P("تومان", className="text-center")
#                     ])
#                 )
#             ], style={"height": "100%"})
#         ], md=6)
#     ])
# ], fluid=True)


# # --- Callback حالا با کلیک روی دکمه فعال می‌شود ---
# @callback(
#     Output('predicted-price-text', 'children'),
#     Input('predict-price-btn', 'n_clicks'),  # ورودی اصلی: کلیک روی دکمه
#     [
#         State('reg-brand', 'value'),      # وضعیت فعلی منوی برند
#         State('reg-category', 'value')    # وضعیت فعلی منوی دسته‌بندی
#     ],
#     prevent_initial_call=True  # جلوگیری از اجرای خودکار در بارگذاری اولیه
# )
# def update_price_prediction(n_clicks, brand, category):
#     """
#     این تابع با کلیک روی دکمه، قیمت را بر اساس آخرین مقادیر انتخاب شده پیش‌بینی می‌کند.
#     """
#     try:
#         # ۱. بررسی ورودی‌های کاربر
#         if not all([brand, category]):
#             return "لطفا برند و دسته‌بندی را انتخاب کنید."

#         # ۲. پیش‌پردازش داده‌ها
#         processed_input = shared_data.preprocess_for_regression(brand, category)

#         # ۳. بررسی نتیجه پیش‌پردازش
#         if processed_input is None:
#             return "خطا در پیش‌پردازش"

#         # ۴. پیش‌بینی با مدل
#         price_pred_log = shared_data.rf_model.predict(processed_input)
#         price_pred = np.expm1(price_pred_log)[0]

#         # ۵. نمایش نتیجه نهایی
#         return f"{price_pred:,.0f}"

#     except Exception as e:
#         shared_data.logger.error(f"--- !!! CALLBACK FAILED: {e} !!! ---", exc_info=True)
#         return "خطا"
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import shared_data 
from shared_data import logger

# --- لایه ظاهری صفحه ---
layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("پیش‌بینی قیمت محصول", className="text-center my-4"), width=12)
    ]),
    dbc.Row([
        # ستون ورودی‌ها
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("ویژگی‌های محصول را انتخاب کنید"),
                dbc.CardBody([
                    dbc.Form([
                        dbc.Label("برند:", html_for="reg-brand", className="mt-3"),
                        dcc.Dropdown(
                            id='reg-brand',
                            options=[{'label': b, 'value': b} for b in shared_data.brands],
                            value=None,
                            placeholder="یک برند را انتخاب کنید...",
                            searchable=True
                        ),
                        dbc.Label("دسته‌بندی:", html_for="reg-category", className="mt-3"),
                        dcc.Dropdown(
                            id='reg-category',
                            options=[{'label': c, 'value': c} for c in shared_data.categories],
                            value=None,
                            placeholder="یک دسته‌بندی را انتخاب کنید...",
                            searchable=True
                        ),
                        dbc.Button("پیش‌بینی قیمت", id="predict-price-btn", color="primary", className="mt-4 w-100"),
                    ])
                ])
            ])
        ], md=6),
        # ستون خروجی
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("نتیجه پیش‌بینی"),
                dbc.CardBody([
                    html.Div(id='price-prediction-output', children=[
                        html.H4("قیمت پیش‌بینی شده:", className="text-center"),
                        html.H2("...", id="predicted-price-text", className="text-center blue text-success fw-bold"),
                        html.P("تومان", className="text-center", id="price-unit")
                    ])
                ])
            ], style={"height": "100%"})
        ], md=6)
    ])
], fluid=True)


# --- Callback تشخیصی ---
@callback(
    Output('predicted-price-text', 'children'),
    Output('predicted-price-text', 'className'),
    Output('price-unit', 'hidden'),
    Input('predict-price-btn', 'n_clicks'),
    State('reg-brand', 'value'),
    State('reg-category', 'value'),
    prevent_initial_call=True
)
def update_price_prediction(n_clicks, brand, category):
    logger.info("--- [DIAGNOSTIC] New Prediction Request ---")
    logger.info(f"[DIAGNOSTIC] Received Brand: '{brand}' | Received Category: '{category}'")

    if not all([brand, category]):
        return "لطفا هر دو مورد را انتخاب کنید.", "text-center text-warning fw-bold", True

    processed_input = shared_data.preprocess_for_regression(brand, category)

    if processed_input is None:
        logger.error("[DIAGNOSTIC] Preprocessing returned None. Aborting prediction.")
        return "خطا در پیش‌پردازش داده‌ها", "text-center text-danger fw-bold", True
    
    try:
        logger.info(f"[DIAGNOSTIC] Shape of data sent to model: {processed_input.shape}")
        
        price_pred_log = shared_data.rf_model.predict(processed_input)
        logger.info(f"[DIAGNOSTIC] Raw prediction from model (log value): {price_pred_log[0]}")
        
        price_pred = np.expm1(price_pred_log)[0]
        logger.info(f"[DIAGNOSTIC] Final price prediction (after expm1): {price_pred}")

        return f"{price_pred:,.0f}", "text-center text-success fw-bold", False
        
    except Exception as e:
        logger.error(f"--- [DIAGNOSTIC] ERROR DURING PREDICTION: {e} ---", exc_info=True)
        return "خطا در فرآیند پیش‌بینی", "text-center text-danger fw-bold", True

