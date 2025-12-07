# streamlit_app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import re
import math

# ----- Constants / Column names -----
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
COL_FIFO_SELL_ORDER = "FIFO_Sell_Order"  # new column to show sell matching order

# possible FMV column names (after normalization)
FMV_CANDIDATES = [
    "FMV",
    "FMV_31Jan2018",
    "FMV_31_01_2018",
    "FMV_31-01-2018",
    "FMV on 31 Jan 2018",
    "FMV Price on 31-Jan-2018",   # matches "FMV Price on\n31-Jan-2018" after normalization
    "FMV Price on 31 Jan 2018",
]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse any newlines / multiple spaces in column names into a single space
    and strip leading/trailing spaces.
    Example: 'FMV Price on\\n31-Jan-2018' -> 'FMV Price on 31-Jan-2018'
    """
    df.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in df.columns]
    return df


def find_fmv_for_row(row, fmv_col_candidates=FMV_CANDIDATES):
    """
    Return FMV for a row:
      1) Look for an FMV column directly in this row (many candidate names).
      2) If not found, return None (we do not use external mapping).
    """
    for c in fmv_col_candidates:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                continue
    return None


def process_merged_dataframe(
    df: pd.DataFrame,
    apply_grandfather: bool = False,
) -> pd.DataFrame:
    """
    Processor with optional Grandfathering (31-Jan-2018).

    Behavior:
      - Merge is expected to be pre-sorted by Pur. Date ascending (the UI sorts before calling).
      - For each ISIN, BUY lots are consumed FIFO in the order they appear (which reflects purchase-date ordering).
      - SELL rows are processed in ascending Sale Date order.
      - A FIFO_Sell_Order integer is assigned to each SELL row in the order sells were matched.
      - Capital Gain column uses original Pur. Price (unchanged).
      - ST/LT allocation uses adjusted cost (max(orig_buy_price, FMV) when apply_grandfather=True).
      - Adjusted cost is NOT capped at sale price (so negative allocations are allowed).
    """
    df = df.copy()
    df = normalize_column_names(df)

    # detect quantity column
    qty_col_candidates = [COL_QTY, "Qty", "Quantity", "Quantity Sold"]
    qty_col = None
    for c in qty_col_candidates:
        if c in df.columns:
            qty_col = c
            break
    if qty_col is None:
        qty_col = COL_QTY  # fallback name

    # ensure required columns exist
    for col in (COL_SALE_DATE, COL_PUR_DATE, qty_col, COL_SALE_PRICE, COL_PUR_PRICE):
        if col not in df.columns:
            df[col] = pd.NA

    # parse datetimes and numerics
    df[COL_SALE_DATE] = pd.to_datetime(df.get(COL_SALE_DATE), errors="coerce")
    df[COL_PUR_DATE] = pd.to_datetime(df.get(COL_PUR_DATE), errors="coerce")
    df[qty_col] = pd.to_numeric(df.get(qty_col), errors="coerce").fillna(0.0)
    df[COL_SALE_PRICE] = pd.to_numeric(df.get(COL_SALE_PRICE), errors="coerce")
    df[COL_PUR_PRICE] = pd.to_numeric(df.get(COL_PUR_PRICE), errors="coerce")

    # compute holding days (row-level helper)
    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    # initialize outputs (ST/LT/CAP and FIFO sell order)
    df[COL_SHORT] = 0.0
    df[COL_LONG] = 0.0
    df[COL_CAP_GAIN] = 0.0
    df[COL_FIFO_SELL_ORDER] = pd.NA

    # CAPITAL GAIN: always (sale - pur) * qty using original Pur. Price
    cap_gain_list = []
    for _, row in df.iterrows():
        q = float(row.get(qty_col) or 0.0)
        sp = row.get(COL_SALE_PRICE)
        pp = row.get(COL_PUR_PRICE)
        if q == 0 or pd.isna(sp) or pd.isna(pp):
            cap_gain_list.append(0.0)
        else:
            cap_gain_list.append(round((float(sp) - float(pp)) * q, 2))
    df[COL_CAP_GAIN] = cap_gain_list

    # normalize Type column if present (BUY/SELL)
    has_type = "Type" in df.columns
    if has_type:
        df["Type_norm"] = df["Type"].astype(str).str.strip().str.upper()

    # global counter to number sells in the order they are matched across all ISINs
    global_sell_counter = 1

    # ---------- FIFO matching per ISIN using purchase-date ordering ----------
    if has_type:
        # process each ISIN independently
        for isin, group in df.groupby(COL_ISIN, sort=False):
            g = group.copy().reset_index()  # preserve original dataframe index in column 'index'

            # separate buys and sells
            buys = g[g["Type_norm"].str.startswith("B", na=False)].copy()
            sells = g[g["Type_norm"].str.startswith("S", na=False)].copy()

            # Because we want global purchase-date ordering across merged files, buys should be in order
            # they appear in the merged dataframe (we pre-sorted merged by Pur. Date in the UI).
            # But to be safe, sort buys explicitly by Pur. Date ascending, and keep stable ordering for equal dates.
            buys["_pur_date_sort"] = buys[COL_PUR_DATE].fillna(pd.Timestamp("9999-12-31"))
            buys = buys.sort_values(["_pur_date_sort"]).reset_index(drop=True)

            # process sells in ascending Sale Date order
            sells["_sale_date_sort"] = sells[COL_SALE_DATE].fillna(pd.Timestamp("9999-12-31"))
            sells = sells.sort_values(["_sale_date_sort"]).reset_index(drop=True)

            # prepare buy_queue: list of dicts { qty, orig_buy_price, buy_date }
            buy_queue = []
            buy_idx = 0  # index into buys to add eligible buys as sells progress

            # prepare maps to capture ST/LT per sell-row original index
            st_map = {}
            lt_map = {}

            # init maps (so we can write back even if zero)
            for _, srow in sells.iterrows():
                st_map[int(srow["index"])] = 0.0
                lt_map[int(srow["index"])] = 0.0

            # iterate sells in chronological order
            for _, srow in sells.iterrows():
                sell_orig_idx = int(srow["index"])
                sell_qty = float(srow.get(qty_col) or 0.0)
                sell_price = srow.get(COL_SALE_PRICE)
                sell_date = srow.get(COL_SALE_DATE)
                if sell_qty <= 0 or pd.isna(sell_price) or pd.isna(sell_date):
                    # nothing to compute
                    continue

                # Add all buys whose Pur. Date <= this sell's Sale Date (these are eligible)
                # Note: buys with missing Pur. Date were placed last when we sorted merged, so not added here.
                while buy_idx < len(buys):
                    candidate = buys.loc[buy_idx]
                    cand_pur = candidate[COL_PUR_DATE]
                    # If candidate has no Pur. Date treat it as not eligible for earlier sells (we placed them last)
                    if pd.isna(cand_pur):
                        break
                    if cand_pur <= sell_date:
                        lot_qty = float(candidate.get(qty_col) or 0.0)
                        orig_buy_price = float(candidate.get(COL_PUR_PRICE) or 0.0)
                        lot_buy_date = candidate.get(COL_PUR_DATE)
                        if lot_qty > 0:
                            buy_queue.append({
                                "qty": lot_qty,
                                "orig_buy_price": orig_buy_price,
                                "buy_date": lot_buy_date
                            })
                        buy_idx += 1
                    else:
                        break

                qty_to_sell = sell_qty

                # consume buy_queue FIFO
                while qty_to_sell > 0 and len(buy_queue) > 0:
                    lot = buy_queue[0]
                    take_qty = min(qty_to_sell, lot["qty"])

                    # adjusted cost per share for allocation
                    adjusted_cost = lot["orig_buy_price"]
                    row_fmv = find_fmv_for_row(srow)
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)
                    # DO NOT cap adjusted_cost at sale price (we allow negative allocations)

                    gain_for_allocation = (float(sell_price) - adjusted_cost) * take_qty

                    # compute holding days for this lot
                    hd = None
                    if pd.notna(lot["buy_date"]) and pd.notna(sell_date):
                        try:
                            hd = int((sell_date - lot["buy_date"]).days)
                        except Exception:
                            hd = None

                    if hd is not None and hd > LONG_TERM_DAYS:
                        lt_map[sell_orig_idx] += gain_for_allocation
                    else:
                        st_map[sell_orig_idx] += gain_for_allocation

                    # decrease quantities
                    lot["qty"] -= take_qty
                    qty_to_sell -= take_qty

                    # remove lot if exhausted
                    if lot["qty"] <= 0:
                        buy_queue.pop(0)
                    else:
                        buy_queue[0] = lot

                # If some quantity remains unmatched (no eligible buys), fall back to sell-row Pur. Price if present
                if qty_to_sell > 0:
                    fallback_cost = float(srow.get(COL_PUR_PRICE) or 0.0)
                    adjusted_cost = fallback_cost
                    row_fmv = find_fmv_for_row(srow)
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)

                    gain_for_allocation = (float(sell_price) - adjusted_cost) * qty_to_sell

                    # determine holding days using sell-row Pur. Date if available
                    hd = None
                    sell_row_buy_date = srow.get(COL_PUR_DATE)
                    if pd.notna(sell_row_buy_date) and pd.notna(sell_date):
                        try:
                            hd = int((sell_date - sell_row_buy_date).days)
                        except Exception:
                            hd = None

                    if hd is not None and hd > LONG_TERM_DAYS:
                        lt_map[sell_orig_idx] += gain_for_allocation
                    else:
                        st_map[sell_orig_idx] += gain_for_allocation

                    qty_to_sell = 0.0

                # assign FIFO_Sell_Order for this sell (global across ISINs)
                df.loc[sell_orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                global_sell_counter += 1

            # write ST/LT back into dataframe
            for idx_key, v in st_map.items():
                df.loc[idx_key, COL_SHORT] = round(v, 2)
            for idx_key, v in lt_map.items():
                df.loc[idx_key, COL_LONG] = round(v, 2)

        # cleanup helper columns if any
        df = df.drop(columns=["Type_norm", "_pur_date_sort", "_sale_date_sort"], errors="ignore")

    else:
        # simple per-row mode (no Type column)
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

            if pd.notna(hd) and hd > LONG_TERM_DAYS:
                short_val, long_val = 0.0, gain_for_allocation
            else:
                short_val, long_val = gain_for_allocation, 0.0

            short_list.append(round(short_val, 2))
            long_list.append(round(long_val, 2))

        df[COL_SHORT] = short_list
        df[COL_LONG] = long_list

    # format dates for output
    for col in [COL_SALE_DATE, COL_PUR_DATE]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%d/%m/%Y")

    return df


# ----------------- Streamlit UI -----------------
st.set_page_config(
    page_title="Capital Gain Calculator (merged by Pur. Date + FIFO)",
    layout="centered",
)

st.title("Capital Gain Calculator")
st.write(
    "Upload one or more Excel/CSV files. "
    "The app will merge them, sort by Purchase Date (ascending), then apply FIFO matching (by purchase-date) per stock. "
    "Optional grandfathering uses an FMV column present in your uploaded files (e.g. 'FMV Price on\\n31-Jan-2018')."
)

# --- Grandfathering control ---
st.markdown("### Grandfathering (31-Jan-2018)")
apply_grandfather = st.checkbox("Apply Grandfathering (use FMV as on 31-Jan-2018)", value=False)

# --- File uploader ---
uploaded = st.file_uploader(
    "Select trade files to merge & process (Excel/CSV). FMV must be present in these files if you want grandfathering applied.",
    accept_multiple_files=True,
    type=["xlsx", "xls", "csv"],
    key="main_files",
)

if uploaded:
    try:
        dfs = []
        for f in uploaded:
            if f.name.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f, header=0)
            df = normalize_column_names(df)
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)

        # sort merged by Pur. Date ascending so buy ordering is global across files
        merged[COL_PUR_DATE] = pd.to_datetime(merged.get(COL_PUR_DATE), errors="coerce")
        merged[COL_SALE_DATE] = pd.to_datetime(merged.get(COL_SALE_DATE), errors="coerce")
        merged = merged.sort_values(by=COL_PUR_DATE, na_position="last").reset_index(drop=True)

        st.subheader("Merged (sorted by Pur. Date ascending) preview")
        st.dataframe(merged.head(120))

        if st.button("Process & Download Excel"):
            # check FMV presence
            merged_fmv_cols = [c for c in merged.columns if c in FMV_CANDIDATES]
            if apply_grandfather and (not merged_fmv_cols):
                st.warning("Grandfathering selected but no FMV column found; processing will continue without FMV.")

            result_df = process_merged_dataframe(merged, apply_grandfather=apply_grandfather)

            st.subheader("Processed preview (first 120 rows)")
            st.dataframe(result_df.head(120))

            st.markdown("**Processed preview (sells arranged by FIFO matching order)**")
            # show sells arranged in FIFO matching order first, then remaining rows
            sells_sorted = result_df[result_df[COL_FIFO_SELL_ORDER].notna()].sort_values(
                by=[COL_FIFO_SELL_ORDER]
            )
            others = result_df[result_df[COL_FIFO_SELL_ORDER].isna()]
            arranged = pd.concat([sells_sorted, others], ignore_index=True)
            st.dataframe(arranged.head(200))

            # prepare Excel for download (include FIFO_Sell_Order)
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Processed")
                arranged.to_excel(writer, index=False, sheet_name="Processed_Sells_First")
            output.seek(0)

            st.download_button(
                label="Download Merged_with_gains.xlsx",
                data=output,
                file_name="Merged_with_gains.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("No files uploaded yet. Select one or more Excel/CSV files to begin.")
