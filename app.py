import pandas as pd
import geopandas as gpd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from dash.dash_table import DataTable
from sklearn.preprocessing import MinMaxScaler
import os

# === Load Data ===
gdf_madrid = gpd.read_file("data/gdf_madrid.gpkg")


pca_weights = {
    "madrid": {
        "high": 0.262, "high_ratio": 0.104, "pro": 0.285, "pro_ratio": 0.047,
        "eur": 0.285, "eur_ratio": 0.017,
        "new_comp": 0.234, "growth": 0.052, "ml": 0.230, "hp": 0.234,
        "size": 0.149, "profit": 0.101,
        "sociozone": 0.404, "businesszone": 0.483, "comp": 0.113
    }
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


def create_region_tab(region, df, geo, weights):
    prefix = region

    sociodemo_keys = [("High", "high"), ("High Ratio", "high_ratio"),
                      ("Professionals", "pro"), ("Professionals Ratio", "pro_ratio"),
                      ("Europeans", "eur"), ("Europeans Ratio", "eur_ratio")]

    business_keys = [("New Companies", "new_comp"), ("Growth Rate", "growth"),
                     ("Medium/Large Companies", "ml"), ("High Profit Companies", "hp"),
                     ("Size Ratio", "size"), ("Profit Ratio", "profit")]

    zone_keys = [("Sociodemographic Score", "sociozone"),
                 ("Business Score", "businesszone"),
                 ("Competitor Score", "comp")]

    return dcc.Tab(label=region.capitalize(), children=html.Div([
        html.H4(f"{region.capitalize()} - Zone Score", style={"fontWeight": "bold"}),

        html.Hr(),
        html.H4("1. Choose Sociodemographic and Business Variables Weights (each category must sum 1)", style={"marginTop": "20px"}),
        manual_inputs_block("Sociodemographic Weights", sociodemo_keys, prefix, weights),
        manual_inputs_block("Business Weights", business_keys, prefix, weights),

        html.Hr(),
        html.H4("2. Choose Sociodemographic, Business and Competitor Score Weights (must sum 1)", style={"marginTop": "30px"}),
        manual_inputs_block("Zone Score Weights", zone_keys, prefix, weights),

        html.Hr(),
        html.Div(id=f"{prefix}_warning", style={"color": "red", "fontWeight": "bold", "marginTop": 20}),
        html.H4("Zone Score Map", style={"marginTop": "30px"}),
        dcc.Graph(id=f"{prefix}_map"),
        html.H4("Top 10 Municipalities", style={"marginTop": "30px"}),
        html.Div(id=f"{prefix}_table", style={"marginBottom": "50px"})
    ]))


app.layout = html.Div([
    html.H1("Interactive Dashboard: Top Municipalities in Madrid and Cataluña", style={"marginBottom": "20px"}),
    
    html.P("""
        This is an interactive dashboard designed to identify the top 10 municipalities to open new bank branches in Madrid and Cataluña based 
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

    html.Hr(style={"margin": "30px 0"}),

    dcc.Tabs([
        create_region_tab("madrid", df_madrid, geo_madrid, pca_weights["madrid"])
        # create_region_tab("cataluna", df_cataluna, geo_cataluna, pca_weights["cataluna"])
    ])
])


def register_callback(region, df, geo):
    @app.callback(
        Output(f"{region}_map", "figure"),
        Output(f"{region}_warning", "children"),
        Output(f"{region}_table", "children"),
        [Input(f"{region}_w_{k}", "value") for k in [
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

        sociodemo_sum = sum(w[k] for k in ["high", "high_ratio", "pro", "pro_ratio", "eur", "eur_ratio"])
        business_sum = sum(w[k] for k in ["new_comp", "growth", "ml", "hp", "size", "profit"])
        zone_total = w["sociozone"] + w["businesszone"] + w["comp"]

        if abs(sociodemo_sum - 1.0) > 0.01:
            return {}, f"Sociodemographic weights must sum to 1.0. Currently: {sociodemo_sum:.2f}", None
        if abs(business_sum - 1.0) > 0.01:
            return {}, f"Business weights must sum to 1.0. Currently: {business_sum:.2f}", None
        if abs(zone_total - 1.0) > 0.01:
            return {}, f"Zone Score weights must sum to 1.0. Currently: {zone_total:.2f}", None

        df_copy = df.copy()

        df_copy["sociodemo_score"] = (
            w["high"] * df_copy["high_score"] +
            w["high_ratio"] * df_copy["high_ratio_score"] +
            w["pro"] * df_copy["professionals_score"] +
            w["pro_ratio"] * df_copy["professionals_ratio_score"] +
            w["eur"] * df_copy["europeans_score"] +
            w["eur_ratio"] * df_copy["europeans_ratio_score"]
        )

        df_copy["business_score"] = (
            w["new_comp"] * df_copy["NumNewCompanies_10S_score"] +
            w["growth"] * df_copy["growth_rate_10S_score"] +
            w["ml"] * df_copy["NumCompanies_ML_10S_score"] +
            w["hp"] * df_copy["NumCompanies_HP_10S_score"] +
            w["size"] * df_copy["companies_size_ratio_10S_score"] +
            w["profit"] * df_copy["companies_profit_ratio_10S_score"]
        )

        df_copy[["sociodemo_score_norm", "business_score_norm", "competitor_score_norm"]] = MinMaxScaler().fit_transform(
            df_copy[["sociodemo_score", "business_score", "competitor_score"]]
        )

        df_copy["zone_score"] = (
            w["sociozone"] * df_copy["sociodemo_score_norm"] +
            w["businesszone"] * df_copy["business_score_norm"] +
            w["comp"] * df_copy["competitor_score_norm"]
        )

        df_copy["top_10"] = df_copy["zone_score"].rank(method="min", ascending=False) <= 10

        gdf = geo.merge(df_copy[[ "municipality", "municipality_name", "zone_score",
                                  "sociodemo_score_norm", "business_score_norm", "competitor_score_norm", "top_10" ]],
                        on="municipality", how="left")

        gdf["color"] = gdf["top_10"].map({True: "Top 10", False: "Others"})

        fig = px.choropleth_map(
            gdf,
            geojson=gdf.geometry,
            locations=gdf.index,
            color="color",
            color_discrete_map={"Top 10": "red", "Others": "lightgrey"},
            map_style=none,
            center={"lat": 40.5, "lon": -3.7},
            zoom=8,
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

        table_df = df_copy[df_copy["top_10"]].sort_values("zone_score", ascending=False)[[
            "municipality", "municipality_name", "sociodemo_score_norm",
            "business_score_norm", "competitor_score_norm", "zone_score"
        ]].copy()
        
        table_df["sociodemo_score_norm"] = table_df["sociodemo_score_norm"].round(3)
        table_df["business_score_norm"] = table_df["business_score_norm"].round(3)
        table_df["competitor_score_norm"] = table_df["competitor_score_norm"].round(3)
        table_df["zone_score"] = table_df["zone_score"].round(3)

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

register_callback("madrid", df_madrid, geo_madrid)
# register_callback("cataluna", df_cataluna, geo_cataluna)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))

