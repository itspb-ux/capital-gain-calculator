# streamlit_app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import re

# ----- Constants / Column names -----
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

# possible FMV candidate column names (after normalization)
FMV_CANDIDATES = [
    "FMV",
    "FMV_31Jan2018",
    "FMV_31_01_2018",
    "FMV_31-01-2018",
    "FMV on 31 Jan 2018",
    "FMV Price on 31-Jan-2018",
    "FMV Price on 31 Jan 2018",
]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse newlines/multiple spaces in column names and trim."""
    df.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in df.columns]
    return df


def find_fmv_for_row(row, fmv_col_candidates=FMV_CANDIDATES):
    """
    Look for an FMV value in any of the known candidate column names in this row.
    Return float or None.
    """
    for c in fmv_col_candidates:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                continue
    return None


def process_merged_dataframe(df: pd.DataFrame, apply_grandfather: bool = False) -> pd.DataFrame:
    """
    Process the merged dataframe and apply FIFO per ISIN while preserving group order.
    - df: merged dataframe (should already be sorted globally by Pur. Date ascending).
    - apply_grandfather: if True, uses FMV per row (if present) when computing adjusted cost for allocation.
    Returns dataframe with Short Term, Long Term, Capital Gain, FIFO_Sell_Order, ISIN_Completion_Order.
    """
    df = df.copy()
    df = normalize_column_names(df)

    # detect quantity column (accept multiple common names)
    qty_candidates = [COL_QTY, "Qty", "Quantity", "Quantity Sold"]
    qty_col = next((c for c in qty_candidates if c in df.columns), COL_QTY)

    # ensure required columns exist
    for c in (COL_ISIN, COL_TYPE, COL_SALE_DATE, COL_PUR_DATE, qty_col, COL_SALE_PRICE, COL_PUR_PRICE):
        if c not in df.columns:
            df[c] = pd.NA

    # parse types/dates/numerics
    df[COL_TYPE] = df.get(COL_TYPE).astype(str).fillna("").str.strip()
    df["Type_norm"] = df[COL_TYPE].str.upper()
    df[COL_SALE_DATE] = pd.to_datetime(df.get(COL_SALE_DATE), errors="coerce")
    df[COL_PUR_DATE] = pd.to_datetime(df.get(COL_PUR_DATE), errors="coerce")
    df[qty_col] = pd.to_numeric(df.get(qty_col), errors="coerce").fillna(0.0)
    df[COL_SALE_PRICE] = pd.to_numeric(df.get(COL_SALE_PRICE), errors="coerce")
    df[COL_PUR_PRICE] = pd.to_numeric(df.get(COL_PUR_PRICE), errors="coerce")

    # helper: holding days per row
    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    # init output columns
    df[COL_SHORT] = 0.0
    df[COL_LONG] = 0.0
    df[COL_CAP_GAIN] = 0.0
    df[COL_FIFO_SELL_ORDER] = pd.NA
    df[COL_ISIN_COMPLETE] = pd.NA

    # compute Capital Gain (always using original Pur. Price)
    cap_gain = []
    for _, row in df.iterrows():
        q = float(row.get(qty_col) or 0.0)
        sp = row.get(COL_SALE_PRICE)
        pp = row.get(COL_PUR_PRICE)
        if q == 0 or pd.isna(sp) or pd.isna(pp):
            cap_gain.append(0.0)
        else:
            cap_gain.append(round((float(sp) - float(pp)) * q, 2))
    df[COL_CAP_GAIN] = cap_gain

    # We will process each ISIN in the order they appear in the merged df (groupby with sort=False preserves this)
    global_sell_counter = 1
    isin_completion_map = {}  # isin -> completion global sell order when fully sold
    isin_bought_qty_map = {}  # isin -> cumulative bought qty (for completion tracking)

    for isin, group in df.groupby(COL_ISIN, sort=False):
        g = group.copy().reset_index()  # 'index' column stores original df index values
        buy_queue = []  # list of dicts: {"qty", "orig_buy_price", "buy_date"}
        cumulative_bought = 0.0
        cumulative_sold = 0.0
        completed_flag = False

        # iterate rows in the group in the merged order (which is now purchase-date-sorted globally)
        for _, row in g.iterrows():
            orig_idx = int(row["index"])
            rtype = str(row["Type_norm"] or "").upper()
            qty = float(row.get(qty_col) or 0.0)
            sale_price = row.get(COL_SALE_PRICE)
            buy_price = row.get(COL_PUR_PRICE)
            sale_date = row.get(COL_SALE_DATE)
            buy_date = row.get(COL_PUR_DATE)

            # FMV for this row if present in the uploaded data
            row_fmv = find_fmv_for_row(row)

            if rtype.startswith("B"):
                # enqueue buy lot (appearance order)
                if qty > 0:
                    orig_buy = float(buy_price) if pd.notna(buy_price) else 0.0
                    buy_queue.append({
                        "qty": qty,
                        "orig_buy_price": orig_buy,
                        "buy_date": buy_date
                    })
                    cumulative_bought += qty

            elif rtype.startswith("S"):
                # perform FIFO consumption from earlier buy_queue only
                qty_to_sell = qty
                if qty_to_sell <= 0 or pd.isna(sale_price):
                    # still assign FIFO sell order (even for zero) and update sold qty
                    df.loc[orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                    global_sell_counter += 1
                    cumulative_sold += qty_to_sell
                    # possible completion check
                    if (not completed_flag) and cumulative_bought > 0 and cumulative_sold >= cumulative_bought:
                        isin_completion_map[isin] = df.loc[orig_idx, COL_FIFO_SELL_ORDER]
                        completed_flag = True
                    continue

                # consume buys FIFO
                remaining = qty_to_sell
                while remaining > 0 and len(buy_queue) > 0:
                    lot = buy_queue[0]
                    take = min(remaining, lot["qty"])

                    adjusted_cost = lot["orig_buy_price"]
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)
                    # DO NOT cap adjusted_cost at sale price => allow negatives

                    alloc_amount = (float(sale_price) - adjusted_cost) * take

                    # determine holding days relative to this lot
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

                # if still remaining (no earlier buys), fallback to this row's pur price (if present)
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

                # assign FIFO sell order
                df.loc[orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                sell_order_this = global_sell_counter
                global_sell_counter += 1

                # completion check for this ISIN
                if (not completed_flag) and cumulative_bought > 0 and cumulative_sold >= cumulative_bought:
                    isin_completion_map[isin] = sell_order_this
                    completed_flag = True

            else:
                # no Type or unknown type: we do not perform FIFO matching on this row; allocation remains as-is
                pass

        # after finishing this ISIN group, mark ISIN_Completion_Order (same for all rows of that ISIN)
        if isin in isin_completion_map:
            df.loc[df[COL_ISIN] == isin, COL_ISIN_COMPLETE] = isin_completion_map[isin]
        else:
            df.loc[df[COL_ISIN] == isin, COL_ISIN_COMPLETE] = pd.NA

    # format dates for output (human-readable)
    for col in (COL_SALE_DATE, COL_PUR_DATE):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%d/%m/%Y")

    return df


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Capital Gain Calculator (PurDate + FIFO, assets kept together)", layout="centered")

st.title("Capital Gain Calculator")
st.write(
    "Upload Excel/CSV files. The app will merge them, sort globally by Purchase Date (ascending), "
    "then process each asset (ISIN) in that merged order and apply strict FIFO (buys consumed in that order). "
    "All rows of each ISIN are kept together in the final output. Optional grandfathering uses an FMV column present in the uploaded files."
)

# --- Grandfathering control ---
apply_grandfather = st.checkbox("Apply Grandfathering (use FMV as on 31-Jan-2018)", value=False)

# --- File uploader ---
uploaded = st.file_uploader(
    "Select Excel/CSV files to merge & process (FMV must be embedded in these files if you want grandfathering applied).",
    accept_multiple_files=True,
    type=["xlsx", "xls", "csv"],
    key="main_files",
)

if uploaded:
    try:
        # read all files and normalize column names
        dfs = []
        for f in uploaded:
            if f.name.lower().endswith(".csv"):
                d = pd.read_csv(f)
            else:
                d = pd.read_excel(f, header=0)
            d = normalize_column_names(d)
            dfs.append(d)

        # concat preserving uploaded data order but we will sort by purchase date next
        merged = pd.concat(dfs, ignore_index=True)

        # --- Global sort: Purchase Date ascending (missing Pur. Date go last) ---
        merged[COL_PUR_DATE] = pd.to_datetime(merged.get(COL_PUR_DATE), errors="coerce")
        merged[COL_SALE_DATE] = pd.to_datetime(merged.get(COL_SALE_DATE), errors="coerce")
        # sort globally by purchase date ascending (this establishes the buy ordering across all files)
        merged = merged.sort_values(by=[COL_PUR_DATE], na_position="last").reset_index(drop=True)

        st.subheader("Merged (sorted globally by Purchase Date asc) — preview")
        st.dataframe(merged.head(120))

        if st.button("Process & Download Excel"):
            # warn if FMV requested but no FMV columns present
            merged_fmv_cols = [c for c in merged.columns if c in FMV_CANDIDATES]
            if apply_grandfather and (not merged_fmv_cols):
                st.warning("Grandfathering selected but no FMV column found in uploaded files — grandfathering will not be applied.")

            # process with FIFO per ISIN in the merged order
            result_df = process_merged_dataframe(merged, apply_grandfather=apply_grandfather)

            # Keep assets together: sort by each ISIN's earliest Pur.Date, then by the original row order (index) within that ISIN
            isin_min_pur = result_df.groupby(COL_ISIN)[COL_PUR_DATE].apply(
                lambda s: pd.to_datetime(s, errors="coerce").min()
            ).to_dict()
            result_df["ISIN_Min_PurDate"] = result_df[COL_ISIN].map(isin_min_pur)

            # sort by (ISIN_Min_PurDate asc, ISIN asc, original index asc)
            result_df = result_df.sort_values(by=["ISIN_Min_PurDate", COL_ISIN, result_df.index.name or result_df.index], na_position="last").reset_index(drop=True)

            # drop helper
            result_df = result_df.drop(columns=["ISIN_Min_PurDate"], errors="ignore")

            st.subheader("Processed (assets kept together) — preview")
            st.dataframe(result_df.head(200))

            # prepare Excel for download
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Processed")
            output.seek(0)

            st.download_button(
                label="Download Processed_FIFO.xlsx",
                data=output,
                file_name="Processed_FIFO.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("No files uploaded yet. Select one or more Excel/CSV files to begin.")
