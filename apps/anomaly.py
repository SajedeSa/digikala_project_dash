# apps/anomaly.py

from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA

from app import app
import shared_data

# آماده‌سازی داده برای مصورسازی
if shared_data.X_scaled is not None and shared_data.df_out is not None:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(shared_data.X_scaled)
else:
    X_pca = np.array([[]])

layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("تحلیل و شناسایی نظرات مشکوک (ناهنجاری)", className="text-center"))),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("کنترل پنل تحلیل"),
                dbc.CardBody([
                    dbc.Label("مدل تشخیص ناهنجاری:", html_for="anomaly-model-selector"),
                    dcc.Dropdown(
                        id="anomaly-model-selector",
                        options=[
                            {'label': 'Autoencoder Score', 'value': 'ae_score'},
                            {'label': 'Ensemble Score', 'value': 'ensemble_score'},
                            {'label': 'Isolation Forest Score', 'value': 'iso_score'},
                            {'label': 'Local Outlier Factor (LOF) Score', 'value': 'lof_score'},
                            {'label': 'KNN Distance', 'value': 'knn_dist'},
                        ],
                        value='ae_score' # مدل پیش‌فرض
                    ),
                    dbc.Label("حد آستانه ناهنجاری (صدک):", html_for="anomaly-threshold-slider", className="mt-3"),
                    dcc.Slider(id='anomaly-threshold-slider', min=90, max=99, value=95, step=1, marks={i: str(i) for i in range(90, 100)}),
                    html.Div(id="anomaly-summary-text", className="mt-3")
                ])
            ])
        ], md=4),
        dbc.Col(
            dcc.Graph(id='anomaly-scatter-plot', style={'height': '60vh'}),
            md=8
        )
    ]),
    dbc.Row(
        dbc.Col(
            html.Div([
                html.H4("نمونه نظرات تشخیص داده شده به عنوان ناهنجاری", className="mt-4"),
                dbc.Table(id='anomaly-table', bordered=True, striped=True, hover=True)
            ])
        )
    )
], fluid=True)


@app.callback(
    [Output('anomaly-scatter-plot', 'figure'),
     Output('anomaly-summary-text', 'children'),
     Output('anomaly-table', 'children')],
    [Input('anomaly-model-selector', 'value'),
     Input('anomaly-threshold-slider', 'value')]
)
def update_anomaly_view(score_col, threshold_percentile):
    if X_pca.shape[1] == 0 or shared_data.df_out is None:
        fig = go.Figure().add_annotation(text="داده برای نمایش وجود ندارد", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        summary = "داده در دسترس نیست."
        table = []
        return fig, summary, table

    scores = shared_data.df_out[score_col]
    threshold_value = np.nanpercentile(scores, threshold_percentile)
    outliers_mask = scores >= threshold_value

    # ساخت نمودار پراکندگی
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=f'نمره {score_col}')
            ),
            text=shared_data.df_out['body'].str[:200] + '...',
            hoverinfo='text',
            name='نظرات'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X_pca[outliers_mask, 0],
            y=X_pca[outliers_mask, 1],
            mode='markers',
            marker=dict(size=12, color='red', symbol='x'),
            name=f'ناهنجاری (بالاتر از صدک {threshold_percentile})'
        )
    )
    fig.update_layout(
        title=f"پراکندگی نظرات بر اساس PCA و نمره ناهنجاری ({score_col})",
        xaxis_title="مؤلفه اصلی اول (PCA1)",
        yaxis_title="مؤلفه اصلی دوم (PCA2)",
        showlegend=True
    )

    # ساخت متن خلاصه
    num_outliers = outliers_mask.sum()
    total_count = len(shared_data.df_out)
    summary = f"با آستانه صدک {threshold_percentile} (مقدار: {threshold_value:.4f})، تعداد {num_outliers} از {total_count} نظر ({num_outliers/total_count:.2%}) به عنوان ناهنجاری تشخیص داده شد."

    # ساخت جدول نمونه‌ها
    df_anomalies = shared_data.df_out[outliers_mask].sort_values(by=score_col, ascending=False).head(10)
    table_header = [html.Thead(html.Tr([html.Th("متن نظر"), html.Th(f"نمره ({score_col})")]))]
    table_body = [html.Tbody([
        html.Tr([
            html.Td(row['body']),
            html.Td(f"{row[score_col]:.4f}")
        ]) for index, row in df_anomalies.iterrows()
    ])]
    table = table_header + table_body

    return fig, summary, table
