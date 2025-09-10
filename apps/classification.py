from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from app import app
import shared_data

layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("مقایسه عملکرد مدل‌های طبقه‌بندی", className="text-center"))),
    dbc.Row(
        dbc.Col(
            dcc.Graph(
                id='comparison-table-graph',
                figure=go.Figure(
                    data=[go.Table(
                        header=dict(values=['مدل'] + list(shared_data.comparison_df.columns), fill_color='paleturquoise', align='left'),
                        cells=dict(values=[shared_data.comparison_df.index] + [shared_data.comparison_df[col] for col in shared_data.comparison_df.columns], fill_color='lavender', align='left')
                    )]
                ).update_layout(title="جدول مقایسه متریک‌های مدل‌ها")
            ) if shared_data.comparison_df is not None else html.Div("داده مقایسه مدل‌ها یافت نشد.")
        )
    ),
    
    dbc.Row(dbc.Col(html.Hr(), className="mt-4")),
    
    dbc.Row(dbc.Col(html.H2("پیش‌بینی پیشنهاد یک نظر", className="text-center mt-4"))),
    dbc.Row([
        dbc.Col([
             dbc.Card([
                dbc.CardHeader("یک نظر را از دیتاست انتخاب کنید"),
                dbc.CardBody([
                    dbc.Label("انتخاب نظر بر اساس متن:", html_for="class-body-dropdown"),
                    dcc.Dropdown(
                        id='class-body-dropdown',
                        options=[
                            {'label': f"{body[:100]}...", 'value': idx} 
                            for idx, body in shared_data.df_preprocessed['body'].head(1000).items() # نمایش ۱۰۰۰ نظر اول برای انتخاب
                        ],
                        placeholder="یک نظر را برای تست انتخاب کنید..."
                    ),
                    html.P("نظر انتخاب شده:", className="mt-3 fw-bold"),
                    html.P(id="selected-comment-text", style={"maxHeight": "100px", "overflowY": "auto", "border": "1px solid #ccc", "padding": "10px", "borderRadius": "5px"})
                ])
             ])
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("نتیجه طبقه‌بندی"),
                dbc.CardBody([
                    html.Div(id='classification-output', children=[
                        html.H4("وضعیت پیشنهاد:", className="text-center"),
                        html.H2("...", id="predicted-class-text", className="text-center fw-bold"),
                    ])
                ])
            ], style={"height": "100%"})
        ], md=6)
    ])
], fluid=True)

@app.callback(
    [Output('predicted-class-text', 'children'),
     Output('predicted-class-text', 'className'),
     Output('selected-comment-text', 'children')],
    Input('class-body-dropdown', 'value'),
    prevent_initial_call=True
)
def update_classification(selected_index):
    if selected_index is None:
        return "...", "text-center fw-bold", "نظری انتخاب نشده است."
        
    if shared_data.best_classifier is not None and shared_data.X_scaled is not None:
        try:
            features = shared_data.X_scaled[selected_index].reshape(1, -1)
            prediction = shared_data.best_classifier.predict(features)[0]
            
            if shared_data.label_encoder:
                class_name = shared_data.label_encoder.inverse_transform([prediction])[0]
            else: # در غیر این صورت، باینری ۰ و ۱ است
                class_name = "پیشنهاد می‌شود" if prediction == 1 else "پیشنهاد نمی‌شود"

            color = "text-success" if (prediction == 1 or "می‌شود" in str(class_name)) else "text-danger"
            selected_text = shared_data.df_preprocessed.loc[selected_index, 'body']
            
            return class_name, f"text-center fw-bold {color}", selected_text
            
        except Exception as e:
            return f"خطا: {e}", "text-center fw-bold text-warning", "خطا در پردازش."
            
    return "مدل در دسترس نیست.", "text-center fw-bold text-warning", "..."