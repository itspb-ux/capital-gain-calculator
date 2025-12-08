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


def looks_like_buy(val: str) -> bool:
    if not val:
        return False
    v = str(val).strip().upper()
    return v.startswith("B") or v.startswith("BUY")


def looks_like_sell(val: str) -> bool:
    if not val:
        return False
    v = str(val).strip().upper()
    return v.startswith("S") or v.startswith("SELL")


# ---------------- FIFO / ALLOCATION ENGINE ----------------

def process_merged_dataframe(df: pd.DataFrame, apply_grandfather: bool = False) -> pd.DataFrame:
    """
    Robust processor:
      - If Type column with BUY/SELL-like values exists, perform FIFO on rows that are BUY/SELL.
      - Rows without BUY/SELL (or when Type absent) are allocated using per-row logic (based on holding days).
      - Capital Gain is always (Sale Price - Pur. Price) * Qty (unchanged).
      - FMV (grandfathering) is taken only from columns in the uploaded files (candidate names).
      - Dates remain datetimes (no string formatting).
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

    # parsing and normalization
    df[COL_TYPE] = df.get(COL_TYPE).astype(str).fillna("").str.strip()
    df["Type_norm"] = df[COL_TYPE].str.upper()
    df[COL_PUR_DATE] = pd.to_datetime(df.get(COL_PUR_DATE), errors="coerce")
    df[COL_SALE_DATE] = pd.to_datetime(df.get(COL_SALE_DATE), errors="coerce")
    df[qty_col] = pd.to_numeric(df.get(qty_col), errors="coerce").fillna(0.0)
    df[COL_SALE_PRICE] = pd.to_numeric(df.get(COL_SALE_PRICE), errors="coerce")
    df[COL_PUR_PRICE] = pd.to_numeric(df.get(COL_PUR_PRICE), errors="coerce")

    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    # initialize output columns
    df[COL_SHORT] = 0.0
    df[COL_LONG] = 0.0
    df[COL_CAP_GAIN] = 0.0
    df[COL_FIFO_SELL_ORDER] = pd.NA
    df[COL_ISIN_COMPLETE] = pd.NA

    # CAPITAL GAIN calculation (always original Pur. Price)
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

    # Determine if we have any clear BUY/SELL-like values in Type column
    type_series = df.get(COL_TYPE, pd.Series([], dtype="object")).astype(str)
    has_buy_sell_like = (
        type_series.str.strip().apply(lambda v: looks_like_buy(v) or looks_like_sell(v)).any()
    )

    # If no buy/sell-like values, do simple per-row allocation for all rows (no FIFO)
    if not has_buy_sell_like:
        short_list = []
        long_list = []
        for _, row in df.iterrows():
            q = float(row.get(qty_col) or 0.0)
            sp = row.get(COL_SALE_PRICE)
            pp = row.get(COL_PUR_PRICE)
            hd = row.get("Holding Days")

            if q == 0 or pd.isna(sp) or pd.isna(pp):
                short_list.append(0.0)
                long_list.append(0.0)
                continue

            row_fmv = find_fmv_for_row(row)
            adjusted_cost = float(pp)
            if apply_grandfather and row_fmv is not None:
                adjusted_cost = max(adjusted_cost, row_fmv)

            gain_for_allocation = (float(sp) - adjusted_cost) * q

            # allocate to ST or LT based on holding days; negative values allowed
            if pd.notna(hd) and hd > LONG_TERM_DAYS:
                short_val, long_val = 0.0, gain_for_allocation
            else:
                short_val, long_val = gain_for_allocation, 0.0

            short_list.append(round(short_val, 2))
            long_list.append(round(long_val, 2))

        df[COL_SHORT] = short_list
        df[COL_LONG] = long_list
        return df

    # ---------- FIFO processing when Type column contains BUY/SELL-like rows ----------
    global_sell_counter = 1
    isin_completion = {}

    # process per ISIN, preserving merged order (groupby sort=False)
    for isin, group in df.groupby(COL_ISIN, sort=False):
        g = group.copy().reset_index()  # 'index' holds original df index
        buy_queue = []  # list of {"qty","orig_buy_price","buy_date"}
        cumulative_bought = 0.0
        cumulative_sold = 0.0
        completed = False

        for _, row in g.iterrows():
            orig_idx = int(row["index"])
            qty = float(row.get(qty_col) or 0.0)
            rtype_raw = row.get(COL_TYPE)
            rtype = str(rtype_raw).strip().upper() if pd.notna(rtype_raw) else ""
            sale_price = row.get(COL_SALE_PRICE)
            buy_price = row.get(COL_PUR_PRICE)
            sale_date = row.get(COL_SALE_DATE)
            buy_date = row.get(COL_PUR_DATE)
            row_fmv = find_fmv_for_row(row)

            # If this row looks like BUY
            if looks_like_buy(rtype):
                if qty > 0:
                    buy_queue.append({
                        "qty": qty,
                        "orig_buy_price": float(buy_price) if pd.notna(buy_price) else 0.0,
                        "buy_date": buy_date
                    })
                    cumulative_bought += qty

            # If this row looks like SELL
            elif looks_like_sell(rtype):
                qty_to_sell = qty
                if qty_to_sell <= 0 or pd.isna(sale_price):
                    # still record sell order and update sold qty (even if zero)
                    df.loc[orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                    global_sell_counter += 1
                    cumulative_sold += qty_to_sell
                    if not completed and cumulative_bought > 0 and cumulative_sold >= cumulative_bought:
                        isin_completion[isin] = df.loc[orig_idx, COL_FIFO_SELL_ORDER]
                        completed = True
                    continue

                remaining = qty_to_sell

                # consume earlier buys FIFO (only existing queue)
                while remaining > 0 and len(buy_queue) > 0:
                    lot = buy_queue[0]
                    take = min(remaining, lot["qty"])

                    adjusted_cost = lot["orig_buy_price"]
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)

                    alloc_amount = (float(sale_price) - adjusted_cost) * take

                    # determine holding days relative to the lot's buy_date
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

                # if any unmatched remaining qty, fallback to this row's Pur. Price
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

                # assign global FIFO sell order
                df.loc[orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                sell_order_now = global_sell_counter
                global_sell_counter += 1

                # completion check
                if not completed and cumulative_bought > 0 and cumulative_sold >= cumulative_bought:
                    isin_completion[isin] = sell_order_now
                    completed = True

            else:
                # Row doesn't look like BUY/SELL — apply per-row allocation for this row (so it won't remain zero)
                q = qty
                sp = sale_price
                pp = buy_price
                hd = None
                if pd.notna(sp) and pd.notna(pp):
                    row_fmv = row_fmv  # obtained earlier
                    adjusted_cost = float(pp)
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)

                    alloc_amount = (float(sp) - adjusted_cost) * q
                    if pd.notna(sale_date) and pd.notna(buy_date):
                        try:
                            hd = int((sale_date - buy_date).days)
                        except Exception:
                            hd = None

                    if pd.notna(hd) and hd > LONG_TERM_DAYS:
                        df.loc[orig_idx, COL_LONG] = round(df.loc[orig_idx, COL_LONG] + alloc_amount, 2)
                    else:
                        df.loc[orig_idx, COL_SHORT] = round(df.loc[orig_idx, COL_SHORT] + alloc_amount, 2)
                # no change to cumulative_bought/sold (we treat this as standalone)

        # write ISIN completion order to all rows of that ISIN (if any)
        df.loc[df[COL_ISIN] == isin, COL_ISIN_COMPLETE] = isin_completion.get(isin, pd.NA)

    return df


# --------------------- UI ---------------------

st.set_page_config(page_title="Capital Gain Calculator (FIFO, robust Type handling)", layout="centered")
st.title("Capital Gain Calculator (FIFO, robust Type handling)")

apply_grandfather = st.checkbox("Apply Grandfathering (FMV on 31-Jan-2018)", value=False)

uploaded = st.file_uploader("Upload trade files (Excel/CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=True)

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

        st.subheader("Merged (original file/row order)")
        st.dataframe(merged.head(120))

        if st.button("Process & Download"):
            result = process_merged_dataframe(merged, apply_grandfather)

            # Final sort: keep ISIN groups together, inside group sort by Sale Date ascending (then original order)
            result["_orig"] = result.index
            result[COL_SALE_DATE] = pd.to_datetime(result[COL_SALE_DATE], errors="coerce")

            result = result.sort_values(by=[COL_ISIN, COL_SALE_DATE, "_orig"], na_position="last").reset_index(drop=True)
            result = result.drop(columns=["_orig"], errors="ignore")

            st.subheader("Final Output (Assets together, Sale Date ASC within each asset)")
            st.dataframe(result.head(200))

            # Export to Excel preserving datetime cell type
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl", datetime_format="DD-MMM-YYYY") as writer:
                result.to_excel(writer, index=False, sheet_name="Processed")
            output.seek(0)

            st.download_button("Download Processed FIFO.xlsx", data=output, file_name="Processed_FIFO.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload files to begin.")
