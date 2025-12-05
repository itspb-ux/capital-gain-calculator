# streamlit_app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import re

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
    df.columns = [
        re.sub(r"\s+", " ", str(col)).strip()
        for col in df.columns
    ]
    return df


def find_fmv_for_row(row, fmv_col_candidates=FMV_CANDIDATES):
    """
    Return FMV for a row:
      1) Look for an FMV column directly in this row (many candidate names).
      2) If not found, return None (we no longer accept an external mapping).
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

    FIFO matching:
      - For each ISIN, BUY rows are considered in ascending Pur. Date order,
        but a BUY is only eligible to match a SELL if its Pur. Date <= that SELL's Sale Date.
      - SELL rows are processed in ascending Sale Date order.
    Important:
      - Capital Gain column uses original Pur. Price (unchanged by grandfathering).
      - ST/LT allocation uses adjusted cost (max(orig_buy_price, FMV)) when grandfathering
        is enabled; adjusted cost is NOT capped at sale price (so negative allocations allowed).
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
        qty_col = COL_QTY  # create/assume

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

    # holding days (row-level, for simple mode)
    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    # output columns: initialize
    df[COL_SHORT] = 0.0
    df[COL_LONG] = 0.0

    # IMPORTANT: compute Capital Gain (unchanged by grandfathering) using original Pur. Price
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

    # normalize Type column if present
    has_type = "Type" in df.columns
    if has_type:
        df["Type_norm"] = df["Type"].astype(str).str.strip().str.upper()

    # ---------- FIFO by purchase-date mode (Type column exists) ----------
    if has_type:
        # Process per ISIN
        for isin, group in df.groupby(COL_ISIN, sort=False):
            g = group.copy().reset_index()  # keep original index in 'index'
            # Separate buys and sells
            buys = g[g["Type_norm"].str.startswith("B", na=False)].copy()
            sells = g[g["Type_norm"].str.startswith("S", na=False)].copy()

            # Normalize date columns for sorting; treat missing Pur. Date as very early (so available)
            # and missing Sale Date sells will be ignored (can't process).
            buys["_pur_date_sort"] = buys[COL_PUR_DATE].fillna(pd.Timestamp("1900-01-01"))
            sells["_sale_date_sort"] = sells[COL_SALE_DATE].fillna(pd.Timestamp("9999-12-31"))

            # sort buys by Pur. Date ascending (FIFO by purchase date)
            buys = buys.sort_values("_pur_date_sort").reset_index(drop=True)
            # sort sells by Sale Date ascending (we process sells chronologically)
            sells = sells.sort_values("_sale_date_sort").reset_index(drop=True)

            # buy_queue holds lots that are eligible for matching (dicts with qty, orig_buy_price, buy_date)
            buy_queue = []
            buy_idx = 0  # index into buys dataframe for adding eligible buys

            # maps to accumulate ST/LT per original row index
            st_map = {}
            lt_map = {}

            # initialize maps for all sell rows (so we can write back later even if zero)
            for _, srow in sells.iterrows():
                st_map[int(srow["index"])] = 0.0
                lt_map[int(srow["index"])] = 0.0

            # iterate through sells in chronological order
            for _, srow in sells.iterrows():
                sell_orig_idx = int(srow["index"])
                sell_qty = float(srow.get(qty_col) or 0.0)
                sell_price = srow.get(COL_SALE_PRICE)
                sell_date = srow.get(COL_SALE_DATE)
                if sell_qty <= 0 or pd.isna(sell_price) or pd.isna(sell_date):
                    # nothing to compute; keep zeros
                    continue

                # Add buys whose Pur. Date <= this sell's Sale Date (or with missing Pur. Date)
                while buy_idx < len(buys):
                    candidate = buys.loc[buy_idx]
                    cand_pur = candidate[COL_PUR_DATE]
                    # treat missing Pur. Date as available immediately (we set to 1900 earlier)
                    if pd.isna(cand_pur) or cand_pur <= sell_date:
                        # push into queue
                        orig_buy_price = float(candidate.get(COL_PUR_PRICE) or 0.0)
                        lot_qty = float(candidate.get(qty_col) or 0.0)
                        lot_buy_date = candidate.get(COL_PUR_DATE)
                        if lot_qty > 0:
                            buy_queue.append({
                                "qty": lot_qty,
                                "orig_buy_price": orig_buy_price,
                                "buy_date": lot_buy_date
                            })
                        buy_idx += 1
                    else:
                        # next candidate buy has pur_date > current sell_date, so stop adding for this sell
                        break

                qty_to_sell = sell_qty

                # consume queued buys FIFO
                while qty_to_sell > 0 and len(buy_queue) > 0:
                    lot = buy_queue[0]
                    take_qty = min(qty_to_sell, lot["qty"])

                    # adjusted cost for allocation (apply grandfathering if requested)
                    adjusted_cost = lot["orig_buy_price"]
                    row_fmv = find_fmv_for_row(srow)
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)
                    # NOTE: we DO NOT cap adjusted_cost at sale price — allow negative allocations

                    gain_for_allocation = (float(sell_price) - adjusted_cost) * take_qty

                    # holding days per lot (sell_date - lot buy_date)
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

                    # reduce lot qty and remaining sell qty
                    lot["qty"] -= take_qty
                    qty_to_sell -= take_qty

                    # pop lot if fully consumed
                    if lot["qty"] <= 0:
                        buy_queue.pop(0)
                    else:
                        buy_queue[0] = lot  # update remaining qty

                # If still qty_to_sell remains, try to include buys that have pur_date > sell_date? NO — we follow purchase-date rule.
                # So unmatched qty falls back to using sell-row pur_price if present, else cost=0
                if qty_to_sell > 0:
                    orig_cost_basis = float(srow.get(COL_PUR_PRICE) or 0.0)
                    adjusted_cost = orig_cost_basis
                    row_fmv = find_fmv_for_row(srow)
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)
                    # allocate remaining
                    gain_for_allocation = (float(sell_price) - adjusted_cost) * qty_to_sell

                    # determine holding days: use sell-row Pur. Date vs Sale Date if present
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

            # write back ST/LT to df using original indices
            for idx_key, v in st_map.items():
                df.loc[idx_key, COL_SHORT] = round(v, 2)
            for idx_key, v in lt_map.items():
                df.loc[idx_key, COL_LONG] = round(v, 2)

        df = df.drop(columns=["Type_norm", "_tx_date"], errors="ignore")

    # ---------- Simple per-row mode (no Type column) ----------
    else:
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

            # FMV for this row if any (only from uploaded data)
            row_fmv = find_fmv_for_row(row)

            # For ST/LT allocation, compute adjusted cost (if grandfathering) else use pp
            adjusted_cost = float(pp)
            if apply_grandfather and row_fmv is not None:
                adjusted_cost = max(adjusted_cost, row_fmv)

            # NOTE: DO NOT cap adjusted_cost at sale price — allow (sp - adjusted_cost) to be negative
            gain_for_allocation = (float(sp) - adjusted_cost) * q

            # allow negative ST/LT values (losses)
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
    page_title="Capital Gain Calculator (with Grandfathering, purchase-date FIFO)",
    layout="centered",
)

st.title("Capital Gain Calculator")
st.write(
    "Upload one or more Excel/CSV files. "
    "The app will merge them, apply FIFO (BUY/SELL) matching by Purchase Date (Pur. Date) and compute Short-Term and Long-Term capital gains. "
    "Optionally, apply the Grandfathering clause using an FMV column present in your uploaded files (e.g. 'FMV Price on\\n31-Jan-2018')."
)

# --- Grandfathering controls ---
st.markdown("### Grandfathering (31-Jan-2018)")
apply_grandfather = st.checkbox(
    "Apply Grandfathering (use FMV as on 31-Jan-2018)",
    value=False,
)

# --- Main file uploader ---
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
        st.subheader("Merged preview")
        st.dataframe(merged.head(50))

        if st.button("Process & Download Excel"):
            # check if merged files themselves contain FMV columns
            merged_fmv_cols = [c for c in merged.columns if c in FMV_CANDIDATES]
            if apply_grandfather and (not merged_fmv_cols):
                st.warning(
                    "Grandfathering is selected but no FMV column was found in the uploaded files. "
                    "Processing will continue WITHOUT applying FMV (grandfathering won't be applied)."
                )

            result_df = process_merged_dataframe(
                merged,
                apply_grandfather=apply_grandfather,
            )

            st.subheader("Processed preview")
            st.dataframe(result_df.head(50))

            # prepare Excel for download
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Processed")
            output.seek(0)

            st.download_button(
                label="Download Merged_with_gains.xlsx",
                data=output,
                file_name="Merged_with_gains.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )
    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("No files uploaded yet. Select one or more Excel/CSV files to begin.")
