from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app
# سرور را برای استفاده در WSGI استخراج می‌کنیم
server = app.server

# وارد کردن صفحات داشبورد
from apps import regression, classification, anomaly

# --- لایه اصلی برنامه ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("پیش‌بینی قیمت", href="/apps/regression")),
            dbc.NavItem(dbc.NavLink("طبقه‌بندی نظرات", href="/apps/classification")),
            dbc.NavItem(dbc.NavLink("تشخیص نظرات مشکوک", href="/apps/anomaly")),
        ],
        brand="داشبورد تحلیل نظرات دیجی‌کالا",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    html.Div(id='page-content', children=[])
])

# --- Callback برای مدیریت مسیریابی ---
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/apps/regression':
        return regression.layout
    if pathname == '/apps/classification':
        return classification.layout
    if pathname == '/apps/anomaly':
        return anomaly.layout
    else:
        # صفحه اصلی
        return html.Div([
            dbc.Container([
                html.H1("به داشبورد هوشمند تحلیل نظرات خوش آمدید", className="display-3"),
                html.P("از منوی بالا یکی از قابلیت‌ها را انتخاب کنید.", className="lead"),
            ], fluid=True, className="py-3"),
        ], className="p-3 bg-light rounded-3")

# --- بخش اصلی برای اجرای برنامه ---
if __name__ == '__main__':
    # اجرای ساده سرور روی پورت 8050
    app.run(debug=True, port=8050)

