from pathlib import Path
import pandas as pd

EXCEL_MAX_DATA_ROWS = 1_048_575  # Excel's sheet limit is 1,048,576 rows, one of which is the header

# ----------------------------------------------------------------------------------------------------
# EXCEL COLOR-CODING
# ----------------------------------------------------------------------------------------------------
def step_07_excel_colours(feature_df, output_path="./MLproduction/production_results_coloured.xlsx"):
    output_path = Path(output_path)

    feature_df = feature_df.copy() # Prevents mutating original dataframe.

    # yhd_kiiht is metadata and never went through data cleaning: it may hold comma decimals or text.
    feature_df["yhd_kiiht"] = pd.to_numeric(
        feature_df["yhd_kiiht"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    # Ratio is undefined when yhd_kiiht is 0 or missing -> leave the cell empty (inf can't be written to Excel).
    denominator = feature_df["yhd_kiiht"].where(feature_df["yhd_kiiht"] != 0)

    # Calculate ratios. See, which type of acceleration is most dominant.
    feature_df["pysty_vs_yhdistetty"] = (feature_df["pys_kiiht"] / denominator).round(3)
    feature_df["sivu_vs_yhdistetty"] = (feature_df["siv_kiiht"] / denominator).round(3)
    feature_df["nyökkimis_vs_yhdistetty"] = (feature_df["nyo_kiiht"] / denominator).round(3)

    # Move 'yhd_kiiht' column to the rightmost position
    cols = list(feature_df.columns)
    if "yhd_kiiht" in cols:
        cols.append(cols.pop(cols.index("yhd_kiiht")))
        feature_df = feature_df[cols]


    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colour_columns = [
        "pysty_vs_yhdistetty", "sivu_vs_yhdistetty", "nyökkimis_vs_yhdistetty",  # ride-ratios
        "yhd_kiiht", "pys_kiiht", "siv_kiiht", "nyo_kiiht",                      # ride-values
    ]
    blue_columns = ['ura_max', 'harjanne_ka', 'kaltevuus', 'rms_mega_oik', 'delta', 'tl332_paapak', 'anomaly_score']

    # Green-yellow-red scale points (min / median / max) from the WHOLE dataset, so a colour means
    # the same value on every sheet. Excel's own min/percentile/max would be computed per sheet.
    scales = {}
    for col_name in colour_columns:
        if col_name in feature_df.columns:
            values = pd.to_numeric(feature_df[col_name], errors="coerce")
            if values.notna().any():
                scales[col_name] = (float(values.min()), float(values.median()), float(values.max()))

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        blue_fill = workbook.add_format({'bg_color': '#ADD8E6'})  # light blue

        # A sheet holds at most EXCEL_MAX_DATA_ROWS rows: split the data into Data, Data_2, Data_3, ...
        for sheet_number, start in enumerate(range(0, len(feature_df), EXCEL_MAX_DATA_ROWS), start=1):
            chunk = feature_df.iloc[start:start + EXCEL_MAX_DATA_ROWS]
            sheet_name = "Data" if sheet_number == 1 else f"Data_{sheet_number}"
            chunk.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            last_row = len(chunk)  # number of rows on this sheet

            for col_name, (low, mid, high) in scales.items():
                col_idx = feature_df.columns.get_loc(col_name)
                worksheet.conditional_format(1, col_idx, last_row, col_idx, {
                    'type': '3_color_scale',
                    'min_type': 'num', 'min_value': low,
                    'mid_type': 'num', 'mid_value': mid,
                    'max_type': 'num', 'max_value': high,
                    'min_color': "#00FF33",   # green
                    'mid_color': "#FFDB4B",   # yellow
                    'max_color': "#FF001E",   # red
                })

            # Add blue fill to specific columns for readability
            for col_name in blue_columns:
                if col_name in feature_df.columns:
                    col_idx = feature_df.columns.get_loc(col_name)
                    # Apply solid fill from row 1 (first data row) to last_row
                    worksheet.set_column(col_idx, col_idx, None, blue_fill)
                    worksheet.conditional_format(1, col_idx, last_row, col_idx, {
                        'type': 'no_blanks',
                        'format': blue_fill
                    })