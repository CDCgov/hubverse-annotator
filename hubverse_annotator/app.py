"""
A streamlit application for that loads hubverse formatted
tables and plots model forecasts for the user to compare
and annotate models.

To run: poetry run streamlit run app.py
"""

import datetime
import logging
import time

import forecasttools
import polars as pl
import streamlit as st

PYRENEW_MODELS = {
    "CFA_Pyrenew-Pyrenew_HE_COVID": "HE_COVID",
    "CFA_Pyrenew-Pyrenew_H_COVID": "H_COVID",
    "CFA_Pyrenew-Pyrenew_HW_COVID": "HW_COVID",
}


# def model_forecast_chart(model_hubverse_table, quantiles):
#     """
#     Given the forecast data in a hubverse table with
#     mul and a sorted list of
#     quantiles, create and return a list of Altair chart
#     layers representing a band for each symmetric pair of
#     quantiles.
#     """
#     layers = []
#     n = len(quantiles)
#     n_pairs = n // 2

#     for i in range(n_pairs):
#         lower = quantiles[i]
#         upper = quantiles[-(i + 1)]
#         opacity = 0.2 + 0.1 * (n_pairs - i - 1)

#         band = (
#             alt.Chart(model_hubverse_table)
#             .transform_filter(
#                 (alt.datum.output_type == "quantile")
#                 & (
#                     (alt.datum.output_type_id == lower)
#                     | (alt.datum.output_type_id == upper)
#                 )
#             )
#             .transform_aggregate(
#                 lower="min(value)",
#                 upper="max(value)",
#                 groupby=["model", "target_end_date"],
#             )
#             .mark_area(color="blue", opacity=opacity)
#             .encode(
#                 x=alt.X("target_end_date:T", title="Target End Date"),
#                 y=alt.Y("lower:Q", title="Forecast Value"),
#                 y2="upper:Q",
#                 detail="model:N",
#             )
#         )

#         layers.append(band)
#     return layers


def main() -> None:
    # initiate logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # record start time
    start_time = time.time()
    # begin streamlit application
    st.title("Forecast Annotator")
    uploaded_file = st.file_uploader(
        "Upload Hubverse File", type=["csv", "parquet"]
    )
    # load the hubverse data
    if uploaded_file is not None:
        if uploaded_file.name.endswith("parquet"):
            smhub_table = pl.read_parquet(uploaded_file)
        else:
            smhub_table = pl.read_csv(uploaded_file)
        logger.info(f"Uploaded file:\n{uploaded_file.name}")
        logger.info(f"Contents\n:{smhub_table}")
        # two-column layout for reference date and location
        col1, col2 = st.columns(2)
        with col1:
            today = datetime.datetime.today().date()
            reference_date = st.date_input("Reference Date", value=today)
        with col2:
            # get locations from forecasttools, some might be excluded from the
            # hubverse table, though
            locations = forecasttools.location_table["long_name"].to_list()
            location = st.selectbox("Location", locations)
        # get location abbreviation
        two_letter_loc_abbr = forecasttools.location_lookup(
            location_vector=[location], location_format="long_name"
        )["location_code"].item()
        # get hubverse table by selected location
        smhub_table = smhub_table.filter(
            pl.col("location") == two_letter_loc_abbr
        )
        # models and targets available
        models_available = smhub_table["model"].unique().to_list()
        selected_models = st.multiselect(
            "Select Models To Plot", options=models_available
        )
        targets_available = smhub_table["target"].unique().to_list()
        selected_target = st.selectbox(
            "Select Targets", options=targets_available
        )
        # filter hubverse table by selected models
        smhub_table = smhub_table.filter(
            pl.col("model").is_in(selected_models),
            pl.col("target") == selected_target,
        )

        st.markdown(f"## Forecasts For: {location}")
        st.markdown(f"## Reference Date: {reference_date}")

        # plotting of the selected model, target, location, and reference date

        # forecasts annotation section
        st.markdown("#### Forecast A")
        st.selectbox(
            "Status", ["Preferred", "Omitted", "None"], key="status_a"
        )
        st.text_input("Comments", key="comments_a")
        st.markdown("#### Forecast B")
        st.selectbox(
            "Status", ["Preferred", "Omitted", "None"], key="status_b"
        )
        st.text_input("Comments", key="comments_b")
        st.markdown("#### Forecast C")
        st.selectbox(
            "Status", ["Preferred", "Omitted", "None"], key="status_c"
        )
        st.text_input("Comments", key="comments_c")
        st.markdown("#### Forecast D")
        st.selectbox(
            "Status", ["Preferred", "Omitted", "None"], key="status_d"
        )
        st.text_input("Comments", key="comments_d")

        # export button
        if st.button("Export forecasts"):
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.success("Need export")
    # record end time
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Session lasted around: {duration // 60} minutes.")


if __name__ == "__main__":
    main()
