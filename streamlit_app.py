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


# ---------------- HELPERS ----------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def find_fmv_for_row(row):
    for c in FMV_CANDIDATES:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                pass
    return None


# ---------------- FIFO ENGINE ----------------

def process_merged_dataframe(df: pd.DataFrame, apply_grandfather: bool = False) -> pd.DataFrame:
    """
    Process merged dataframe:
      - FIFO per ISIN using appearance order (buys appended, sells consume earlier buys).
      - ST/LT allocation uses adjusted cost (max(orig_buy_price, FMV) if grandfathering enabled).
      - Capital Gain always computed as (Sale Price - Pur. Price) * Qty.
      - Returns data frame with ST, LT, CAP_GAIN, FIFO_Sell_Order, ISIN_Completion_Order.
      - IMPORTANT: dates remain as datetime objects (no strftime).
    """
    df = df.copy()
    df = normalize_column_names(df)

    # detect qty column
    qty_candidates = [COL_QTY, "Qty", "Quantity", "Quantity Sold"]
    qty_col = next((c for c in qty_candidates if c in df.columns), COL_QTY)

    # ensure required columns exist
    for c in (COL_ISIN, COL_TYPE, COL_SALE_DATE, COL_PUR_DATE, qty_col, COL_SALE_PRICE, COL_PUR_PRICE):
        if c not in df.columns:
            df[c] = pd.NA

    # parsing
    df[COL_TYPE] = df.get(COL_TYPE).astype(str).fillna("").str.strip()
    df["Type_norm"] = df[COL_TYPE].str.upper()
    df[COL_PUR_DATE] = pd.to_datetime(df.get(COL_PUR_DATE), errors="coerce")
    df[COL_SALE_DATE] = pd.to_datetime(df.get(COL_SALE_DATE), errors="coerce")
    df[qty_col] = pd.to_numeric(df.get(qty_col), errors="coerce").fillna(0.0)
    df[COL_SALE_PRICE] = pd.to_numeric(df.get(COL_SALE_PRICE), errors="coerce")
    df[COL_PUR_PRICE] = pd.to_numeric(df.get(COL_PUR_PRICE), errors="coerce")

    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    # outputs initialization
    df[COL_SHORT] = 0.0
    df[COL_LONG] = 0.0
    df[COL_CAP_GAIN] = 0.0
    df[COL_FIFO_SELL_ORDER] = pd.NA
    df[COL_ISIN_COMPLETE] = pd.NA

    # Capital gain = (sale - pur) * qty (uses original Pur. Price)
    capgains = []
    for _, row in df.iterrows():
        q = float(row.get(qty_col) or 0.0)
        sp = row.get(COL_SALE_PRICE)
        pp = row.get(COL_PUR_PRICE)
        if q and pd.notna(sp) and pd.notna(pp):
            capgains.append(round((float(sp) - float(pp)) * q, 2))
        else:
            capgains.append(0.0)
    df[COL_CAP_GAIN] = capgains

    # FIFO per ISIN in grouped order (appearance order preserved by groupby sort=False)
    global_sell_counter = 1
    isin_completion = {}

    for isin, group in df.groupby(COL_ISIN, sort=False):
        g = group.copy().reset_index()  # original index in 'index'
        buy_queue = []
        cumulative_bought = 0.0
        cumulative_sold = 0.0
        completed = False

        for _, row in g.iterrows():
            orig_idx = int(row["index"])
            rtype = row["Type_norm"]
            qty = float(row.get(qty_col) or 0.0)
            sale_price = row.get(COL_SALE_PRICE)
            buy_price = row.get(COL_PUR_PRICE)
            sale_date = row.get(COL_SALE_DATE)
            buy_date = row.get(COL_PUR_DATE)
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
                if qty_to_sell <= 0 or pd.isna(sale_price):
                    # still assign FIFO sell order (even if zero) and update sold qty
                    df.loc[orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                    global_sell_counter += 1
                    cumulative_sold += qty_to_sell
                    if not completed and cumulative_bought > 0 and cumulative_sold >= cumulative_bought:
                        isin_completion[isin] = df.loc[orig_idx, COL_FIFO_SELL_ORDER]
                        completed = True
                    continue

                remaining = qty_to_sell

                # consume earlier buys FIFO
                while remaining > 0 and len(buy_queue) > 0:
                    lot = buy_queue[0]
                    take = min(remaining, lot["qty"])

                    adjusted_cost = lot["orig_buy_price"]
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)

                    alloc_amount = (float(sale_price) - adjusted_cost) * take

                    # compute holding days relative to lot buy date
                    hd = None
                    if pd.notna(lot["buy_date"]) and pd.notna(sale_date):
                        try:
                            hd = int((sale_date - lot["buy_date"]).days)
                        except Exception:
                            hd = None

                    if hd is not None and hd > LONG_TERM_DAYS:
                        df.loc[orig_idx, COL_LONG] = round(df.loc[orig_idx, COL_LONG] + alloc_amount, 2)
                    else:
                        df.loc[orig_idx, COL_SHORT] = round(df.loc[orig_idx, COL_SHORT] + alloc_amount, 2)

                    lot["qty"] -= take
                    remaining -= take
                    cumulative_sold += take

                    if lot["qty"] <= 0:
                        buy_queue.pop(0)
                    else:
                        buy_queue[0] = lot

                # any unmatched qty falls back to sell-row Pur. Price
                if remaining > 0:
                    fallback_cost = float(buy_price) if pd.notna(buy_price) else 0.0
                    adjusted_cost = fallback_cost
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)

                    alloc_amount = (float(sale_price) - adjusted_cost) * remaining

                    hd = None
                    if pd.notna(buy_date) and pd.notna(sale_date):
                        try:
                            hd = int((sale_date - buy_date).days)
                        except Exception:
                            hd = None

                    if hd is not None and hd > LONG_TERM_DAYS:
                        df.loc[orig_idx, COL_LONG] = round(df.loc[orig_idx, COL_LONG] + alloc_amount, 2)
                    else:
                        df.loc[orig_idx, COL_SHORT] = round(df.loc[orig_idx, COL_SHORT] + alloc_amount, 2)

                    cumulative_sold += remaining
                    remaining = 0.0

                # assign FIFO sell order (global sequence)
                df.loc[orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                sell_order_now = global_sell_counter
                global_sell_counter += 1

                # completion check
                if not completed and cumulative_bought > 0 and cumulative_sold >= cumulative_bought:
                    isin_completion[isin] = sell_order_now
                    completed = True

            else:
                # no Type / unknown: skip FIFO matching (row left as-is)
                pass

        # write completion order for all rows of this ISIN
        df.loc[df[COL_ISIN] == isin, COL_ISIN_COMPLETE] = isin_completion.get(isin, pd.NA)

    # IMPORTANT: we keep date columns as datetimes (do not convert to strings)
    # Caller/UI will format/display as needed; Excel writer will be instructed to keep datetime format.

    return df


# --------------------- UI ---------------------

st.set_page_config(page_title="Capital Gain Calculator (FIFO, Sale-Date Ordering)", layout="centered")

st.title("Capital Gain Calculator (FIFO by Appearance, Sort by Sale Date)")

apply_grandfather = st.checkbox("Apply Grandfathering (FMV on 31-Jan-2018)", value=False)

uploaded = st.file_uploader("Upload trade files", type=["xlsx", "xls", "csv"], accept_multiple_files=True)

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

        st.subheader("Merged (Original Row Order)")
        st.dataframe(merged.head(100))

        if st.button("Process & Download"):
            result = process_merged_dataframe(merged, apply_grandfather)

            # FINAL SORT: keep ISIN groups together; inside each ISIN sort by Sale Date ascending (then original order)
            result["_orig_order"] = result.index
            result[COL_SALE_DATE] = pd.to_datetime(result[COL_SALE_DATE], errors="coerce")

            result = result.sort_values(
                by=[COL_ISIN, COL_SALE_DATE, "_orig_order"],
                na_position="last"
            ).reset_index(drop=True)

            result = result.drop(columns=["_orig_order"], errors="ignore")

            st.subheader("Final Output (Assets Together, Sale-Date ASC)")
            st.dataframe(result.head(200))

            # Export to Excel while preserving datetime cells (Excel will see proper dates)
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl", datetime_format="DD-MMM-YYYY") as writer:
                result.to_excel(writer, index=False, sheet_name="Processed")
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
    st.info("Upload files to start.")
