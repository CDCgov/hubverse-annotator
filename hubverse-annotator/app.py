"""
A streamlit application for that loads hubverse formatted
tables and plots model forecasts for the user to compare
and annotate models.

To run: poetry run streamlit run app.py
"""

import datetime

import altair as alt
import forecasttools
import polars as pl
import streamlit as st

PYRENEW_MODELS = {
    "CFA_Pyrenew-Pyrenew_HE_COVID": "HE_COVID",
    "CFA_Pyrenew-Pyrenew_H_COVID": "H_COVID",
    "CFA_Pyrenew-Pyrenew_HW_COVID": "HW_COVID",
}


def create_quantile_bands(data, quantiles):
    """
    Given the data (e.g. smhub_table) and a sorted list of
    quantiles, create and return a list of Altair chart
    layers representing a band for each symmetric pair of
    quantiles.
    """
    layers = []
    n = len(quantiles)
    n_pairs = n // 2

    for i in range(n_pairs):
        lower = quantiles[i]
        upper = quantiles[-(i + 1)]
        opacity = 0.2 + 0.1 * (n_pairs - i - 1)

        band = (
            alt.Chart(data)
            .transform_filter(
                (alt.datum.output_type == "quantile")
                & (
                    (alt.datum.output_type_id == lower)
                    | (alt.datum.output_type_id == upper)
                )
            )
            .transform_aggregate(
                lower="min(value)",
                upper="max(value)",
                groupby=["model", "target_end_date"],
            )
            .mark_area(color="blue", opacity=opacity)
            .encode(
                x=alt.X("target_end_date:T", title="Target End Date"),
                y=alt.Y("lower:Q", title="Forecast Value"),
                y2="upper:Q",
                detail="model:N",
            )
        )

        layers.append(band)
    return layers


def main() -> None:
    st.title("Forecast Annotator")
    uploaded_file = st.file_uploader(
        "Upload Hubverse File", type=["csv", "parquet"]
    )
    # two-column layout for reference date and location
    col1, col2 = st.columns(2)
    with col1:
        today = datetime.datetime.today().date()
        # reference date might have to come from uploaded hubverse table
        reference_date = st.date_input("Reference Date", value=today)
    with col2:
        location = st.selectbox(
            "Location", ["Arizona", "New York", "Nevada", "New Jersey"]
        )
    # get location abbreviation
    two_letter_loc_abbr = forecasttools.location_lookup(
        location_vector=[location], location_format="long_name"
    )["location_code"].item()
    # load the hubverse data
    if uploaded_file is not None:
        if uploaded_file.name.endswith("parquet"):
            smhub_table = pl.read_parquet(uploaded_file)
        else:
            smhub_table = pl.read_csv(uploaded_file)
        smhub_table = smhub_table.filter(
            pl.col("location") == two_letter_loc_abbr
        )
        # setup area chart
        # chart = alt.Chart(
        #     smhub_table
        # ).mark_line(point=True).encode(
        #     x=alt.X("target_end_date:T", title="Target End Date"),
        #     y=alt.Y("value:Q", title="Forecast Value"),
        #     color=alt.Color("model:N", title="Model")
        # ).properties(
        #     title="Forecasts For Pyrenew-HEW Models",
        # )
        # setup area chart
        chart = (
            alt.Chart(smhub_table)
            .transform_filter(alt.datum.output_type == "quantile")
            .transform_aggregate(
                lower="min(value)",
                upper="max(value)",
                groupby=["model", "target_end_date"],
            )
            .mark_area(color="blue", opacity=0.2)
            .encode(
                x=alt.X("target_end_date:T", title="Target End Date"),
                y=alt.Y("value:"),
            )
        )
        # show on streamlit
        st.altair_chart(chart)
    st.markdown(f"## Forecasts For: {location}")
    st.markdown(f"## Reference Date: {reference_date}")

    # forecasts annotation section
    st.markdown("#### Forecast A")
    st.selectbox("Status", ["Preferred", "Omitted", "None"], key="status_a")
    st.text_input("Comments", key="comments_a")

    st.markdown("#### Forecast B")
    st.selectbox("Status", ["Preferred", "Omitted", "None"], key="status_b")
    st.text_input("Comments", key="comments_b")

    st.markdown("#### Forecast C")
    st.selectbox("Status", ["Preferred", "Omitted", "None"], key="status_c")
    st.text_input("Comments", key="comments_c")

    st.markdown("#### Forecast D")
    st.selectbox("Status", ["Preferred", "Omitted", "None"], key="status_d")
    st.text_input("Comments", key="comments_d")

    # export button
    if st.button("Export forecasts"):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.success("Need export")


if __name__ == "__main__":
    main()
