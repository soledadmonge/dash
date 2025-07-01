import pandas as pd
import geopandas as gpd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback_context
from sklearn.preprocessing import MinMaxScaler

# === Load Data ===
df_madrid = pd.read_csv("data/madrid.csv")
df_cataluna = pd.read_csv("data/cataluna.csv")
geo_madrid = gpd.read_file("data/municipalities_madrid.geojson")
geo_cataluna = gpd.read_file("data/municipalities_cataluna.geojson")

df_madrid["municipality"] = df_madrid["municipality"].astype(str)
df_cataluna["municipality"] = df_cataluna["municipality"].astype(str).str.zfill(5)
geo_madrid["municipality"] = geo_madrid["municipality"].astype(str)
geo_cataluna["municipality"] = geo_cataluna["municipality"].astype(str)

pca_weights = {
    "madrid": {
        "high": 0.1063, "high_ratio": 0.0421, "pro": 0.1145, "pro_ratio": 0.0187,
        "eur": 0.1148, "eur_ratio": 0.0074,
        "new_comp": 0.1133, "growth": 0.0250, "ml": 0.1113, "hp": 0.1127,
        "size": 0.0720, "profit": 0.0485,
        "sociozone": 0.4038, "businesszone": 0.4827, "comp": 0.1135
    },
    "cataluna": {
        "high": 0.1132, "high_ratio": 0.0173, "pro": 0.1223, "pro_ratio": 0.0515,
        "eur": 0.1229, "eur_ratio": 0.0421,
        "new_comp": 0.1178, "growth": 0.0301, "ml": 0.1165, "hp": 0.1184,
        "size": 0.0153, "profit": 0.0164,
        "sociozone": 0.4694, "businesszone": 0.4146, "comp": 0.1160
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
                    step=0.0001,
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
        html.H2(f"{region.capitalize()} - Zone Score"),
        manual_inputs_block("Sociodemographic Variables", sociodemo_keys, prefix, weights),
        manual_inputs_block("Business Variables", business_keys, prefix, weights),
        manual_inputs_block("Zone Score Weights (Must Sum 1)", zone_keys, prefix, weights),
        html.Div(id=f"{prefix}_warning", style={"color": "red", "fontWeight": "bold", "marginTop": 20}),
        dcc.Graph(id=f"{prefix}_map")
    ]))

app.layout = html.Div([
    dcc.Tabs([
        create_region_tab("madrid", df_madrid, geo_madrid, pca_weights["madrid"]),
        create_region_tab("cataluna", df_cataluna, geo_cataluna, pca_weights["cataluna"])
    ])
])

def register_callback(region, df, geo):
    @app.callback(
        Output(f"{region}_map", "figure"),
        Output(f"{region}_warning", "children"),
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

        # Validations
        sociodemo_sum = sum(w[k] for k in ["high", "high_ratio", "pro", "pro_ratio", "eur", "eur_ratio"])
        business_sum = sum(w[k] for k in ["new_comp", "growth", "ml", "hp", "size", "profit"])
        zone_total = w["sociozone"] + w["businesszone"] + w["comp"]

        if abs(sociodemo_sum - w["sociozone"]) > 0.01:
            return dash.no_update, f"Sociodemographic sub-weights must sum to Sociodemographic Score ({w['sociozone']:.2f}), but total is {sociodemo_sum:.2f}"
        if abs(business_sum - w["businesszone"]) > 0.01:
            return dash.no_update, f"Business sub-weights must sum to Business Score ({w['businesszone']:.2f}), but total is {business_sum:.2f}"
        if abs(zone_total - 1.0) > 0.01:
            return dash.no_update, f"Zone Score weights must sum to 1.0. Currently: {zone_total:.2f}"

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

        # Flag top 10
        df_copy["top_10"] = df_copy["zone_score"].rank(method="min", ascending=False) <= 10

        gdf = geo.merge(df_copy[[
            "municipality", "municipality_name", "zone_score", "sociodemo_score_norm",
            "business_score_norm", "competitor_score_norm", "top_10"
        ]], on="municipality", how="left")

        gdf["color"] = gdf["top_10"].map({True: "Top 10", False: "Others"})


        fig = px.choropleth_mapbox(
            gdf,
            geojson=gdf.geometry,
            locations=gdf.index,
            color="color",
            color_discrete_map={"Top 10": "red", "Others": "lightgrey"},
            mapbox_style="carto-positron",
            center={"lat": 41.5, "lon": 1.5} if region == "cataluna" else {"lat": 40.4, "lon": -3.7},
            zoom=7,
            opacity=0.6,
            hover_name="municipality_name",
            hover_data={
                "zone_score": ":.2f",
                "sociodemo_score_norm": ":.2f",
                "business_score_norm": ":.2f",
                "competitor_score_norm": ":.2f",
                "color": False,
                "top_10": False
            },
            labels={
                "zone_score": "Zone Score",
                "sociodemo_score_norm": "Sociodemographic Score",
                "business_score_norm": "Business Score",
                "competitor_score_norm": "Competitor Score",
                "color": " "
            }

        )

        return fig, ""

register_callback("madrid", df_madrid, geo_madrid)
register_callback("cataluna", df_cataluna, geo_cataluna)

if __name__ == "__main__":
    app.run(debug=True, port=8052)
