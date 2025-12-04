# streamlit_app.py
import streamlit as st
import pandas as pd
from io import BytesIO

# ----- Paste your processing function (re-used/adjusted from your Flask app) -----
COL_ISIN = "ISIN"
COL_SALE_DATE = "Sale Date"
COL_PUR_DATE = "Pur. Date"
COL_QTY = "Qty. Sold"
COL_SALE_PRICE = "Sale Price"
COL_PUR_PRICE = "Pur. Price"
COL_SHORT = "Short Term"
COL_LONG = "Long Term"
COL_CAP_GAIN = "Capital Gain"
LONG_TERM_DAYS = 365

def process_merged_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df[COL_SALE_DATE] = pd.to_datetime(df.get(COL_SALE_DATE), errors="coerce")
    df[COL_PUR_DATE] = pd.to_datetime(df.get(COL_PUR_DATE), errors="coerce")

    df[COL_QTY] = pd.to_numeric(df.get(COL_QTY), errors="coerce")
    df[COL_SALE_PRICE] = pd.to_numeric(df.get(COL_SALE_PRICE), errors="coerce")
    df[COL_PUR_PRICE] = pd.to_numeric(df.get(COL_PUR_PRICE), errors="coerce")

    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    for c in (COL_SHORT, COL_LONG, COL_CAP_GAIN):
        if c not in df.columns:
            df[c] = 0.0

    short_list, long_list, total_list = [], [], []

    for _, row in df.iterrows():
        q = row.get(COL_QTY)
        sp = row.get(COL_SALE_PRICE)
        pp = row.get(COL_PUR_PRICE)
        hd = row.get("Holding Days")

        if pd.isna(q) or pd.isna(sp) or pd.isna(pp):
            short_list.append(0.0); long_list.append(0.0); total_list.append(0.0)
            continue

        gain = (sp - pp) * q

        if pd.notna(hd) and hd > LONG_TERM_DAYS:
            st, lt = 0.0, gain
        else:
            st, lt = gain, 0.0

        short_list.append(round(st, 2))
        long_list.append(round(lt, 2))
        total_list.append(round(gain, 2))

    df[COL_SHORT] = short_list
    df[COL_LONG] = long_list
    df[COL_CAP_GAIN] = total_list

    for col in [COL_SALE_DATE, COL_PUR_DATE]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%d/%m/%Y")

    return df
# ------------------------------------------------------------------------------


st.set_page_config(page_title="Capital Gain Calculator", layout="centered")

st.title("Capital Gain Calculator (Streamlit)")
st.write("Upload one or more Excel files (.xlsx/.xls). They will be merged and the app will calculate Holding Days, Short/Long term gains.")

uploaded = st.file_uploader("Select Excel files", accept_multiple_files=True, type=["xlsx", "xls"])

if uploaded:
    try:
        dfs = []
        for f in uploaded:
            # read with header=0 by default; adjust if your header row is different
            df = pd.read_excel(f, header=0)
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        st.subheader("Merged preview")
        st.dataframe(merged.head(50))

        if st.button("Process & Download Excel"):
            result_df = process_merged_dataframe(merged)

            # show some results in app
            st.subheader("Processed preview")
            st.dataframe(result_df.head(50))

            # prepare in-memory Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Processed")
            output.seek(0)

            st.download_button(
                label="Download Merged_with_gains.xlsx",
                data=output,
                file_name="Merged_with_gains.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("No files uploaded yet. Select one or more Excel files from your computer.")
