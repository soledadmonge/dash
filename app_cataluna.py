import os
import pandas as pd
import geopandas as gpd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from dash.dash_table import DataTable
from sklearn.preprocessing import MinMaxScaler

# === Load Data ===
gdf_cataluna = gpd.read_file("data/gdf_cataluna.gpkg")

pca_weights = {
    "high": 0.241, "high_ratio": 0.036, "pro": 0.260, "pro_ratio": 0.111,
    "eur": 0.262, "eur_ratio": 0.090,
    "new_comp": 0.285, "growth": 0.072, "ml": 0.283, "hp": 0.285,
    "size": 0.036, "profit": 0.039,
    "sociozone": 0.469, "businesszone": 0.415, "comp": 0.116
}

app = Dash(__name__)

def manual_inputs_block(title, keys, prefix, weights):
    return html.Div([
        html.H4(title, style={"marginTop": "25px"}),
        html.Div([
            html.Div([
                html.Label(f"{label}: "),
                dcc.Input(
                    id=f"{prefix}_w_{key}",
                    type="number",
                    value=weights[key],
                    step=0.001,
                    style={"marginRight": "10px", "width": "80px"}
                )
            ], style={"display": "inline-block", "marginRight": "15px"})
            for label, key in keys
        ])
    ])

# === App Layout (sin pestañas) ===
app.layout = html.Div([
    html.H1("Interactive Dashboard: Top Municipalities in Cataluña", style={"marginBottom": "20px"}),

    html.P("""
        This is an interactive dashboard designed to identify the top 10 municipalities to open new bank branches in Cataluña based 
        on sociodemographic, business, and competition characteristics.
    """),
    html.P("""
        - Sociodemographic variables include the number of high-income individuals, professionals, and Europeans 
        (both Spanish and foreign citizens from the European Union). These variables are also represented as ratios, 
        adjusted by the total population of each municipality.
    """),
    html.P("""
        - Business variables include the number of new companies, medium/large companies, and high-profit companies. 
        Additionally, ratio-based variables are included, such as business growth rate and the ratios of medium/large 
        companies and high-profit companies—each relative to the total number of companies in the municipality.
    """),
    html.P("""
        - The competition variable is defined as the inverse of the total number of banks in each municipality. 
        A competition score near 0 indicates high competition (many banks), while a score close to 1 suggests 
        little or no competition.
    """),
    html.P("""
        Finally, each municipality’s Zone Score is calculated as a weighted sum of the Sociodemographic Score, 
        Business Score, and Competitor Score. The default weights, shown below, were obtained using the PCA method, 
        but they can be adjusted directly in this dashboard according to the bank’s strategic preferences.
    """),

    html.Hr(),

    html.H4("1. Choose Sociodemographic and Business Variables Weights (each category must sum 1)", style={"marginTop": "20px"}),

    manual_inputs_block("Sociodemographic Weights", [
        ("High", "high"), ("High Ratio", "high_ratio"),
        ("Professionals", "pro"), ("Professionals Ratio", "pro_ratio"),
        ("Europeans", "eur"), ("Europeans Ratio", "eur_ratio")
    ], "cataluna", pca_weights),

    manual_inputs_block("Business Weights", [
        ("New Companies", "new_comp"), ("Growth Rate", "growth"),
        ("Medium/Large Companies", "ml"), ("High Profit Companies", "hp"),
        ("Size Ratio", "size"), ("Profit Ratio", "profit")
    ], "cataluna", pca_weights),

    html.Hr(),

    html.H4("2. Choose Sociodemographic, Business and Competitor Score Weights (must sum 1)", style={"marginTop": "30px"}),

    manual_inputs_block("Zone Score Weights", [
        ("Sociodemographic Score", "sociozone"),
        ("Business Score", "businesszone"),
        ("Competitor Score", "comp")
    ], "cataluna", pca_weights),

    html.Hr(),
    html.Div(id="cataluna_warning", style={"color": "red", "fontWeight": "bold", "marginTop": 20}),
    html.H4("Zone Score Map", style={"marginTop": "30px"}),
    dcc.Graph(id="cataluna_map"),
    html.H4("Top 10 Municipalities", style={"marginTop": "30px"}),
    html.Div(id="cataluna_table", style={"marginBottom": "50px"})
])

# === Callback ===
@app.callback(
    Output("cataluna_map", "figure"),
    Output("cataluna_warning", "children"),
    Output("cataluna_table", "children"),
    [Input(f"cataluna_w_{k}", "value") for k in [
        "high", "high_ratio", "pro", "pro_ratio", "eur", "eur_ratio",
        "new_comp", "growth", "ml", "hp", "size", "profit",
        "sociozone", "businesszone", "comp"
    ]]
)
def update_map(*weights):
    keys = ["high", "high_ratio", "pro", "pro_ratio", "eur", "eur_ratio",
            "new_comp", "growth", "ml", "hp", "size", "profit",
            "sociozone", "businesszone", "comp"]
    w = dict(zip(keys, weights))

    # Check weight constraints
    sociodemo_sum = sum(w[k] for k in ["high", "high_ratio", "pro", "pro_ratio", "eur", "eur_ratio"])
    business_sum = sum(w[k] for k in ["new_comp", "growth", "ml", "hp", "size", "profit"])
    zone_total = w["sociozone"] + w["businesszone"] + w["comp"]

    if abs(sociodemo_sum - 1.0) > 0.01:
        return {}, f"Sociodemographic weights must sum to 1.0. Currently: {sociodemo_sum:.2f}", None
    if abs(business_sum - 1.0) > 0.01:
        return {}, f"Business weights must sum to 1.0. Currently: {business_sum:.2f}", None
    if abs(zone_total - 1.0) > 0.01:
        return {}, f"Zone Score weights must sum to 1.0. Currently: {zone_total:.2f}", None

    df = gdf_cataluna.copy()

    df["sociodemo_score"] = (
        w["high"] * df["high_score"] +
        w["high_ratio"] * df["high_ratio_score"] +
        w["pro"] * df["professionals_score"] +
        w["pro_ratio"] * df["professionals_ratio_score"] +
        w["eur"] * df["europeans_score"] +
        w["eur_ratio"] * df["europeans_ratio_score"]
    )

    df["business_score"] = (
        w["new_comp"] * df["NumNewCompanies_10S_score"] +
        w["growth"] * df["growth_rate_10S_score"] +
        w["ml"] * df["NumCompanies_ML_10S_score"] +
        w["hp"] * df["NumCompanies_HP_10S_score"] +
        w["size"] * df["companies_size_ratio_10S_score"] +
        w["profit"] * df["companies_profit_ratio_10S_score"]
    )

    df[["sociodemo_score_norm", "business_score_norm", "competitor_score_norm"]] = MinMaxScaler().fit_transform(
        df[["sociodemo_score", "business_score", "competitor_score"]]
    )

    df["zone_score"] = (
        w["sociozone"] * df["sociodemo_score_norm"] +
        w["businesszone"] * df["business_score_norm"] +
        w["comp"] * df["competitor_score_norm"]
    )

    df["top_10"] = df["zone_score"].rank(method="min", ascending=False) <= 10
    df["color"] = df["top_10"].map({True: "Top 10", False: "Others"})

    fig = px.choropleth_mapbox(
    df,
    geojson=df.geometry,
    locations=df.index,
    color="color",
    color_discrete_map={"Top 10": "red", "Others": "lightgrey"},
    mapbox_style="white-bg",
    center={"lat": 40.55, "lon": -3.69},
    zoom=7,  # Zoom más bajo para ver toda la región
    opacity=0.6,
    hover_name="municipality_name",
    hover_data={
        "zone_score": ":.2f",
        "sociodemo_score_norm": ":.2f",
        "business_score_norm": ":.2f",
        "competitor_score_norm": ":.2f",
        "color": False,
        "top_10": False
    }
)

    # Create top 10 table
    table_df = df[df["top_10"]].sort_values("zone_score", ascending=False)[[
        "municipality", "municipality_name", "sociodemo_score_norm",
        "business_score_norm", "competitor_score_norm", "zone_score"
    ]].copy()

    table_df = table_df.round(3)

    table = DataTable(
        data=table_df.to_dict("records"),
        columns=[
            {"name": "Municipality", "id": "municipality"},
            {"name": "Name", "id": "municipality_name"},
            {"name": "Sociodemographic", "id": "sociodemo_score_norm"},
            {"name": "Business", "id": "business_score_norm"},
            {"name": "Competitor", "id": "competitor_score_norm"},
            {"name": "Zone Score", "id": "zone_score"},
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'padding': '5px', 'textAlign': 'left'},
        style_header={'fontWeight': 'bold', 'backgroundColor': 'lightgrey'}
    )

    return fig, "", table

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
