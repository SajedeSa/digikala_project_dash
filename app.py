import dash
import dash_bootstrap_components as dbc
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    'https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.0.3/Vazirmatn-font-face.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "داشبورد تحلیل نظرات دیجی‌کالا"
server = app.server