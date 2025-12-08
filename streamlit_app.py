# streamlit_app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import re

# ------- CONSTANTS -------
COL_ISIN = "ISIN"
COL_TYPE = "Type"
COL_SALE_DATE = "Sale Date"
COL_PUR_DATE = "Pur. Date"
COL_QTY = "Qty. Sold"
COL_SALE_PRICE = "Sale Price"
COL_PUR_PRICE = "Pur. Price"
COL_SHORT = "Short Term"
COL_LONG = "Long Term"
COL_CAP_GAIN = "Capital Gain"
LONG_TERM_DAYS = 365
COL_FIFO_SELL_ORDER = "FIFO_Sell_Order"
COL_ISIN_COMPLETE = "ISIN_Completion_Order"

FMV_CANDIDATES = [
    "FMV",
    "FMV_31Jan2018",
    "FMV_31_01_2018",
    "FMV_31-01-2018",
    "FMV on 31 Jan 2018",
    "FMV Price on 31-Jan-2018",
    "FMV Price on 31 Jan 2018",
]


# -------------- HELPERS --------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in df.columns]
    return df


def find_fmv_for_row(row):
    for c in FMV_CANDIDATES:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except:
                pass
    return None


# -------------- FIFO PROCESSOR --------------

def process_merged_dataframe(df: pd.DataFrame, apply_grandfather: bool = False) -> pd.DataFrame:

    df = df.copy()
    df = normalize_column_names(df)

    # detect quantity column
    qty_candidates = [COL_QTY, "Qty", "Quantity", "Quantity Sold"]
    qty_col = next((c for c in qty_candidates if c in df.columns), COL_QTY)

    # ensure required columns exist
    for col in (COL_ISIN, COL_TYPE, COL_SALE_DATE, COL_PUR_DATE, qty_col, COL_SALE_PRICE, COL_PUR_PRICE):
        if col not in df.columns:
            df[col] = pd.NA

    # parsing
    df[COL_TYPE] = df.get(COL_TYPE).astype(str).fillna("").str.strip()
    df["Type_norm"] = df[COL_TYPE].str.upper()
    df[COL_PUR_DATE] = pd.to_datetime(df[COL_PUR_DATE], errors="coerce")
    df[COL_SALE_DATE] = pd.to_datetime(df[COL_SALE_DATE], errors="coerce")
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)
    df[COL_SALE_PRICE] = pd.to_numeric(df[COL_SALE_PRICE], errors="coerce")
    df[COL_PUR_PRICE] = pd.to_numeric(df[COL_PUR_PRICE], errors="coerce")

    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    # outputs
    df[COL_SHORT] = 0.0
    df[COL_LONG] = 0.0
    df[COL_CAP_GAIN] = 0.0
    df[COL_FIFO_SELL_ORDER] = pd.NA
    df[COL_ISIN_COMPLETE] = pd.NA

    # Capital Gain (ALWAYS original purchase price)
    caps = []
    for _, row in df.iterrows():
        q = float(row.get(qty_col) or 0.0)
        sp = row.get(COL_SALE_PRICE)
        pp = row.get(COL_PUR_PRICE)
        if q and pd.notna(sp) and pd.notna(pp):
            caps.append(round((float(sp) - float(pp)) * q, 2))
        else:
            caps.append(0.0)
    df[COL_CAP_GAIN] = caps

    global_sell_counter = 1
    isin_completion = {}

    # FIFO per ISIN (in grouped order)
    for isin, group in df.groupby(COL_ISIN, sort=False):
        g = group.copy().reset_index()  # "index" column gives original df index
        buy_queue = []
        cumulative_bought = 0.0
        cumulative_sold = 0.0
        completed = False

        for _, row in g.iterrows():
            orig_idx = int(row["index"])
            qty = float(row[qty_col] or 0.0)
            rtype = row["Type_norm"]
            sale_price = row[COL_SALE_PRICE]
            buy_price = row[COL_PUR_PRICE]
            sale_date = row[COL_SALE_DATE]
            buy_date = row[COL_PUR_DATE]
            row_fmv = find_fmv_for_row(row)

            if rtype.startswith("B"):
                if qty > 0:
                    buy_queue.append({
                        "qty": qty,
                        "orig_buy_price": float(buy_price) if pd.notna(buy_price) else 0.0,
                        "buy_date": buy_date
                    })
                    cumulative_bought += qty

            elif rtype.startswith("S"):
                qty_to_sell = qty

                # allocate FIFO
                remaining = qty_to_sell
                while remaining > 0 and buy_queue:
                    lot = buy_queue[0]
                    take = min(remaining, lot["qty"])

                    adjusted_cost = lot["orig_buy_price"]
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)

                    alloc = (float(sale_price) - adjusted_cost) * take

                    # LT / ST logic
                    hd = None
                    if pd.notna(lot["buy_date"]) and pd.notna(sale_date):
                        hd = int((sale_date - lot["buy_date"]).days)

                    if hd and hd > LONG_TERM_DAYS:
                        df.loc[orig_idx, COL_LONG] += round(alloc, 2)
                    else:
                        df.loc[orig_idx, COL_SHORT] += round(alloc, 2)

                    lot["qty"] -= take
                    remaining -= take
                    cumulative_sold += take

                    if lot["qty"] <= 0:
                        buy_queue.pop(0)

                # if unmatched qty → fallback to Pur.Price
                if remaining > 0:
                    fallback_cost = float(buy_price) if pd.notna(buy_price) else 0.0
                    if apply_grandfather and row_fmv is not None:
                        fallback_cost = max(fallback_cost, row_fmv)

                    alloc = (float(sale_price) - fallback_cost) * remaining

                    hd = None
                    if pd.notna(buy_date) and pd.notna(sale_date):
                        hd = int((sale_date - buy_date).days)

                    if hd and hd > LONG_TERM_DAYS:
                        df.loc[orig_idx, COL_LONG] += round(alloc, 2)
                    else:
                        df.loc[orig_idx, COL_SHORT] += round(alloc, 2)

                    cumulative_sold += remaining

                # assign FIFO sell order
                df.loc[orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                sell_order_now = global_sell_counter
                global_sell_counter += 1

                # completion check
                if not completed and cumulative_sold >= cumulative_bought and cumulative_bought > 0:
                    isin_completion[isin] = sell_order_now
                    completed = True

        # write completion order
        df.loc[df[COL_ISIN] == isin, COL_ISIN_COMPLETE] = isin_completion.get(isin, pd.NA)

    # format dates
    df[COL_SALE_DATE] = pd.to_datetime(df[COL_SALE_DATE], errors="coerce").dt.strftime("%d/%m/%Y")
    df[COL_PUR_DATE] = pd.to_datetime(df[COL_PUR_DATE], errors="coerce").dt.strftime("%d/%m/%Y")

    return df


# ------------------ UI -------------------

st.set_page_config(page_title="Capital Gain Calculator (FIFO)", layout="centered")

st.title("Capital Gain Calculator (FIFO, Assets Grouped)")

apply_grandfather = st.checkbox("Apply Grandfathering (FMV on 31-Jan-2018)", value=False)

uploaded = st.file_uploader(
    "Upload Excel/CSV trade files",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True
)

if uploaded:
    try:
        dfs = []
        for f in uploaded:
            if f.name.lower().endswith(".csv"):
                d = pd.read_csv(f)
            else:
                d = pd.read_excel(f)
            dfs.append(normalize_column_names(d))

        merged = pd.concat(dfs, ignore_index=True)

        # global sort by Purchase Date ASC
        merged[COL_PUR_DATE] = pd.to_datetime(merged.get(COL_PUR_DATE), errors="coerce")
        merged = merged.sort_values(by=[COL_PUR_DATE], na_position="last").reset_index()

        st.subheader("Merged (Purchase Date ASC)")
        st.dataframe(merged.head(100))

        if st.button("Process & Download"):

            result_df = process_merged_dataframe(merged, apply_grandfather)

            # keep assets together → sort by earliest Pur.Date per ISIN + original order

            isin_min_pur = (
                pd.to_datetime(result_df[COL_PUR_DATE], errors="coerce")
                .groupby(result_df[COL_ISIN])
                .min()
                .to_dict()
            )

            result_df["ISIN_Min_Pur"] = result_df[COL_ISIN].map(isin_min_pur)
            result_df["_orig_order"] = result_df.index

            result_df = result_df.sort_values(
                by=["ISIN_Min_Pur", COL_ISIN, "_orig_order"],
                na_position="last"
            ).reset_index(drop=True)

            result_df = result_df.drop(columns=["ISIN_Min_Pur", "_orig_order"], errors="ignore")

            st.subheader("Final FIFO Output (Assets kept together)")
            st.dataframe(result_df.head(200))

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Processed")
            output.seek(0)

            st.download_button(
                "Download Processed FIFO.xlsx",
                data=output,
                file_name="Processed_FIFO.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload files to begin.")
