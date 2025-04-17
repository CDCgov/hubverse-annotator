"""
A streamlit application for that loads hubverse formatted
tables and plots model forecasts for the user to compare
and annotate models.

To run: poetry run streamlit run app.py
"""

import logging
import time

import altair as alt
import forecasttools
import pandas as pd
import polars as pl
import streamlit as st

PYRENEW_MODELS = {
    "CFA_Pyrenew-Pyrenew_HE_COVID": "HE_COVID",
    "CFA_Pyrenew-Pyrenew_H_COVID": "H_COVID",
    "CFA_Pyrenew-Pyrenew_HW_COVID": "HW_COVID",
}


def create_forecast_chart(
    hubverse_table: pl.DataFrame, reference_date: str
) -> alt.LayerChart:
    """
    Ingests a hubverse table for a location, reference
    date, and target, and produces quantile plots.
    """
    # altair chart seems to require a pandas dataframe
    hubverse_pd_df = hubverse_table.to_pandas()
    reference_date = pd.to_datetime(reference_date)
    hubverse_pd_df["target_end_date"] = pd.to_datetime(
        hubverse_pd_df["target_end_date"]
    )
    forecast_data = hubverse_pd_df[
        hubverse_pd_df["target_end_date"] >= reference_date
    ]
    historical_data = hubverse_pd_df[
        hubverse_pd_df["target_end_date"] < reference_date
    ]
    pivot = (
        alt.Chart(forecast_data)
        .transform_filter("datum.output_type == 'quantile'")
        .transform_pivot(
            pivot="output_type_id",
            value="value",
            groupby=["model", "target_end_date"],
        )
    )
    errorband = pivot.mark_errorband(color="blue", opacity=0.5).encode(
        x=alt.X("target_end_date:T", title="Target End Date"),
        y=alt.Y("0.05:Q", title="Forecast Value"),
        y2=alt.Y2("0.95:Q"),
        color=alt.Color("model:N", title="Model"),
    )
    median_line = pivot.mark_line(point=True).encode(
        x="target_end_date:T",
        y=alt.Y("0.50:Q", title="Forecast Value"),
        color="model:N",
    )
    historical_points = (
        alt.Chart(historical_data)
        .mark_circle(size=60, color="darkblue", opacity=0.6)
        .encode(
            x=alt.X("target_end_date:T", title="Target End Date"),
            y=alt.Y("value:Q", title="Forecast Value"),
            tooltip=["model", "target_end_date:T", "value:Q"],
        )
    )
    chart = alt.layer(historical_points, errorband, median_line).properties(
        title="Forecasts with Quantile Bands", width=700, height=400
    )

    return chart


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
            ref_dates_available = (
                smhub_table["reference_date"].unique().to_list()
            )
            selected_ref_date = st.selectbox(
                "Reference Date", options=ref_dates_available
            )
        with col2:
            # get locations from forecasttools, some might be excluded from the
            # hubverse table, though
            locations_available = forecasttools.location_table[
                "long_name"
            ].to_list()
            location = st.multiselect("Location", options=locations_available)

        # get location abbreviation
        two_letter_loc_abbrs = [
            forecasttools.location_lookup(
                location_vector=[loc], location_format="long_name"
            )["location_code"].item()
            for loc in locations_available
        ]
        # get hubverse table by selected location
        smhub_table = smhub_table.filter(
            pl.col("location").is_in(two_letter_loc_abbrs)
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
        # filter hubverse table by selected models and target
        smhub_table = smhub_table.filter(
            pl.col("model").is_in(selected_models),
            pl.col("target") == selected_target,
        )

        st.markdown(f"## Forecasts For: {*locations_available}")
        st.markdown(f"## Reference Date: {selected_ref_date}")

        # plotting of the selected model, target, location, and reference date
        forecast_chart = create_forecast_chart(smhub_table, selected_ref_date)
        print(forecast_chart)
        st.altair_chart(forecast_chart, use_container_width=True)

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

# Notes
# Default to latest reference
# Default to US
# Default to all models
# Calendar picker (just show reference date)
# Still needs w/ re-runs
# Interactive plots (toggle)
