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

    IMPORTANT: COL_CAP_GAIN (Capital Gain) is always computed using the original
    purchase price (Pur. Price) as present in the data. Grandfathering only affects
    how gains/losses are allocated between Short Term and Long Term (COL_SHORT / COL_LONG).
    Negative ST/LT values (losses) are preserved and shown.
    NOTE: adjusted cost for allocation is NOT capped at sale price here (so allocation can be negative).
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
    # For rows where data missing, capgain is 0
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

    # ---------- FIFO mode (Type column exists) ----------
    if has_type:
        for isin, group in df.groupby(COL_ISIN, sort=False):
            g = group.copy().reset_index()  # original row index in 'index'
            # transaction date: sale or purchase date
            g["_tx_date"] = g[COL_SALE_DATE].fillna(g[COL_PUR_DATE])
            g = g.sort_values("_tx_date").reset_index(drop=True)

            # FIFO buy queue
            buy_queue = []  # each: {"qty", "orig_buy_price", "buy_date"}
            st_map = {}
            lt_map = {}

            for _, r in g.iterrows():
                orig_idx = r["index"]
                rtype = str(r.get("Type_norm", "")).upper()
                qty = float(r.get(qty_col) or 0.0)
                sale_price = r.get(COL_SALE_PRICE)
                buy_price = r.get(COL_PUR_PRICE)
                sale_date = r.get(COL_SALE_DATE)
                buy_date = r.get(COL_PUR_DATE)

                st_map[orig_idx] = 0.0
                lt_map[orig_idx] = 0.0

                # FMV for this row/ISIN (if any) — only from uploaded data
                row_fmv = find_fmv_for_row(r)

                if rtype.startswith("B"):  # BUY
                    if qty > 0:
                        # store original buy price separately (orig_buy_price) to preserve Capital Gain calc later
                        orig_buy = float(buy_price) if pd.notna(buy_price) else 0.0
                        buy_queue.append(
                            {
                                "qty": qty,
                                "orig_buy_price": orig_buy,
                                "buy_date": buy_date if pd.notna(buy_date) else pd.NaT,
                            }
                        )

                elif rtype.startswith("S"):  # SELL
                    qty_to_sell = qty
                    if qty_to_sell <= 0 or pd.isna(sale_price) or pd.isna(sale_date):
                        continue

                    # consume existing buys (FIFO). For ST/LT allocation, use adjusted cost if grandfathering is enabled.
                    while qty_to_sell > 0 and len(buy_queue) > 0:
                        lot = buy_queue[0]
                        take_qty = min(qty_to_sell, lot["qty"])

                        # compute adjusted cost for allocation (do NOT cap at sale price so losses remain negative)
                        adjusted_cost = lot["orig_buy_price"]
                        if apply_grandfather and row_fmv is not None:
                            adjusted_cost = max(adjusted_cost, row_fmv)
                        # NOTE: we DO NOT min(adjusted_cost, sale_price) here — allow negative allocation

                        gain_for_allocation = (float(sale_price) - adjusted_cost) * take_qty

                        # holding period per lot
                        hd = None
                        if pd.notna(lot["buy_date"]) and pd.notna(sale_date):
                            try:
                                hd = int((sale_date - lot["buy_date"]).days)
                            except Exception:
                                hd = None

                        # allow negative numbers: allocate gain_for_allocation (can be negative)
                        if hd is not None and hd > LONG_TERM_DAYS:
                            lt_map[orig_idx] += gain_for_allocation
                        else:
                            st_map[orig_idx] += gain_for_allocation

                        lot["qty"] -= take_qty
                        qty_to_sell -= take_qty
                        if lot["qty"] <= 0:
                            buy_queue.pop(0)

                    # unmatched sell qty (no buys) — allocate using row-level buy price (orig)
                    if qty_to_sell > 0:
                        orig_cost_basis = float(buy_price) if pd.notna(buy_price) else 0.0
                        adjusted_cost = orig_cost_basis
                        if apply_grandfather and row_fmv is not None:
                            adjusted_cost = max(adjusted_cost, row_fmv)
                        # DO NOT cap adjusted_cost at sale_price here

                        gain_for_allocation = (float(sale_price) - adjusted_cost) * qty_to_sell

                        hd = None
                        if pd.notna(buy_date) and pd.notna(sale_date):
                            try:
                                hd = int((sale_date - buy_date).days)
                            except Exception:
                                hd = None

                        if hd is not None and hd > LONG_TERM_DAYS:
                            lt_map[orig_idx] += gain_for_allocation
                        else:
                            st_map[orig_idx] += gain_for_allocation

                        qty_to_sell = 0.0

            # write back ST/LT using original indices
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
    page_title="Capital Gain Calculator (with Grandfathering)",
    layout="centered",
)

st.title("Capital Gain Calculator")
st.write(
    "Upload one or more Excel/CSV files. "
    "The app will merge them, apply FIFO (if BUY/SELL available) or simple per-row "
    "calculation, and compute Short-Term and Long-Term capital gains. "
    "Optionally, apply the Grandfathering clause using an FMV column present in your uploaded files (e.g. 'FMV Price on\\n31-Jan-2018')."
)

# --- Grandfathering controls ---
st.markdown("### Grandfathering (31-Jan-2018)")
apply_grandfather = st.checkbox(
    "Apply Grandfathering (use FMV as on 31-Jan-2018)",
    value=False,
)

# --- Main file uploader (no separate FMV mapping) ---
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
