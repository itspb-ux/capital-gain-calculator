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

# possible FMV column names (after normalization)
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
    df.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in df.columns]
    return df


def find_fmv_for_row(row, fmv_col_candidates=FMV_CANDIDATES):
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
    Merge-preserving FIFO matching by appearance order within each ISIN.
    - Rows are processed in the same order they appear in the merged dataframe.
    - For each ISIN: when a BUY row is encountered, push its lot to the FIFO buy_queue.
      When a SELL row is encountered, consume from the buy_queue FIFO (only earlier buys).
    - Capital Gain unchanged (sale - pur) * qty using original Pur. Price.
    - ST/LT allocation uses adjusted cost (max(orig_buy_price, FMV)) if grandfathering enabled.
    - Adjusted cost is NOT capped at sale price (so negative ST/LT allowed).
    """
    df = df.copy()
    df = normalize_column_names(df)

    # detect qty column
    qty_col_candidates = [COL_QTY, "Qty", "Quantity", "Quantity Sold"]
    qty_col = next((c for c in qty_col_candidates if c in df.columns), COL_QTY)

    # ensure required columns exist
    for col in (COL_ISIN, COL_TYPE, COL_SALE_DATE, COL_PUR_DATE, qty_col, COL_SALE_PRICE, COL_PUR_PRICE):
        if col not in df.columns:
            df[col] = pd.NA

    # parse dates/numerics robustly
    df[COL_SALE_DATE] = pd.to_datetime(df.get(COL_SALE_DATE), errors="coerce")
    df[COL_PUR_DATE] = pd.to_datetime(df.get(COL_PUR_DATE), errors="coerce")
    df[qty_col] = pd.to_numeric(df.get(qty_col), errors="coerce").fillna(0.0)
    df[COL_SALE_PRICE] = pd.to_numeric(df.get(COL_SALE_PRICE), errors="coerce")
    df[COL_PUR_PRICE] = pd.to_numeric(df.get(COL_PUR_PRICE), errors="coerce")

    # helper: holding days
    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    # initialize outputs
    df[COL_SHORT] = 0.0
    df[COL_LONG] = 0.0
    df[COL_CAP_GAIN] = 0.0
    df[COL_FIFO_SELL_ORDER] = pd.NA

    # compute capital gain (always using original Pur. Price)
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

    # normalize type column
    df[COL_TYPE] = df.get(COL_TYPE).astype(str).fillna("").str.strip()
    df["Type_norm"] = df[COL_TYPE].str.upper()

    # We'll process per ISIN, but keep the merged order.
    # For each ISIN, iterate rows in merged order; add BUY lots when encountered; consume on SELL.
    global_sell_counter = 1

    for isin, group in df.groupby(COL_ISIN, sort=False):
        g = group.copy().reset_index()  # 'index' column preserves original df indices

        # buy_queue holds dicts: { qty, orig_buy_price, buy_date }
        buy_queue = []

        # maps to accumulate ST/LT per original row index
        st_map = {}
        lt_map = {}

        # initialize maps for rows that are sells (so we can write back later)
        for _, row in g.iterrows():
            if str(row["Type_norm"]).startswith("S"):
                st_map[int(row["index"])] = 0.0
                lt_map[int(row["index"])] = 0.0

        # iterate rows in merged appearance order for this ISIN
        for _, row in g.iterrows():
            orig_idx = int(row["index"])
            rtype = row["Type_norm"]
            qty = float(row.get(qty_col) or 0.0)
            sale_price = row.get(COL_SALE_PRICE)
            buy_price = row.get(COL_PUR_PRICE)
            sale_date = row.get(COL_SALE_DATE)
            buy_date = row.get(COL_PUR_DATE)

            # FMV for this row if present (we apply when allocating for sells)
            row_fmv = find_fmv_for_row(row)

            if rtype.startswith("B"):
                # push buy lot in appearance order
                if qty > 0:
                    orig_buy = float(buy_price) if pd.notna(buy_price) else 0.0
                    buy_queue.append({
                        "qty": qty,
                        "orig_buy_price": orig_buy,
                        "buy_date": buy_date
                    })

            elif rtype.startswith("S"):
                # process sell: consume from buy_queue FIFO (only previous buys)
                qty_to_sell = qty
                if qty_to_sell <= 0 or pd.isna(sale_price):
                    # nothing to do
                    continue

                # ensure maps initialized
                st_map.setdefault(orig_idx, 0.0)
                lt_map.setdefault(orig_idx, 0.0)

                while qty_to_sell > 0 and len(buy_queue) > 0:
                    lot = buy_queue[0]
                    take_qty = min(qty_to_sell, lot["qty"])

                    adjusted_cost = lot["orig_buy_price"]
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)
                    # DO NOT cap adjusted_cost at sale price — allow negative allocations

                    gain_for_allocation = (float(sale_price) - adjusted_cost) * take_qty

                    # compute holding period relative to buy lot
                    hd = None
                    if pd.notna(lot["buy_date"]) and pd.notna(sale_date):
                        try:
                            hd = int((sale_date - lot["buy_date"]).days)
                        except Exception:
                            hd = None

                    if hd is not None and hd > LONG_TERM_DAYS:
                        lt_map[orig_idx] += gain_for_allocation
                    else:
                        st_map[orig_idx] += gain_for_allocation

                    # reduce quantities
                    lot["qty"] -= take_qty
                    qty_to_sell -= take_qty

                    if lot["qty"] <= 0:
                        buy_queue.pop(0)
                    else:
                        buy_queue[0] = lot

                # If still some qty remains (no earlier buys to match), fallback to using this sell row's pur price
                if qty_to_sell > 0:
                    orig_cost_basis = float(buy_price) if pd.notna(buy_price) else 0.0
                    adjusted_cost = orig_cost_basis
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)

                    gain_for_allocation = (float(sale_price) - adjusted_cost) * qty_to_sell

                    # determine holding days using sell-row Pur. Date if present
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

                # assign FIFO sell order in processing sequence
                df.loc[orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                global_sell_counter += 1

            else:
                # row with no Type: treat as per-row simple calculation; no FIFO matching
                # (we already computed Capital Gain earlier)
                pass

        # after processing group, write back ST/LT into original dataframe rows
        for idx_key, val in st_map.items():
            df.loc[idx_key, COL_SHORT] = round(val, 2)
        for idx_key, val in lt_map.items():
            df.loc[idx_key, COL_LONG] = round(val, 2)

    # format dates for output
    for col in [COL_SALE_DATE, COL_PUR_DATE]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%d/%m/%Y")

    return df


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Capital Gain Calculator (FIFO by appearance order)", layout="centered")

st.title("Capital Gain Calculator")
st.write(
    "Upload one or more Excel/CSV files. The app will merge them (preserving file/row order), "
    "and for each asset (ISIN) it will apply strict FIFO matching by appearance order within that asset: "
    "a SELL consumes earlier BUY lots only; the next BUY is used only after earlier lots are fully sold."
)

# Grandfathering control
apply_grandfather = st.checkbox("Apply Grandfathering (use FMV as on 31-Jan-2018)", value=False)

# File uploader
uploaded = st.file_uploader(
    "Select trade files to merge & process (Excel/CSV). FMV column must be present in these files to apply grandfathering.",
    accept_multiple_files=True,
    type=["xlsx", "xls", "csv"],
    key="main_files",
)

if uploaded:
    try:
        dfs = []
        for f in uploaded:
            if f.name.lower().endswith(".csv"):
                d = pd.read_csv(f)
            else:
                d = pd.read_excel(f, header=0)
            d = normalize_column_names(d)
            dfs.append(d)

        # merge preserving upload/file row order
        merged = pd.concat(dfs, ignore_index=True)

        st.subheader("Merged preview (original order preserved)")
        st.dataframe(merged.head(120))

        if st.button("Process & Download Excel"):
            # check for FMV presence if grandfathering requested
            merged_fmv_cols = [c for c in merged.columns if c in FMV_CANDIDATES]
            if apply_grandfather and (not merged_fmv_cols):
                st.warning("Grandfathering selected but no FMV column found; processing will continue without FMV.")

            result_df = process_merged_dataframe(merged, apply_grandfather=apply_grandfather)

            st.subheader("Processed preview")
            st.dataframe(result_df.head(200))

            # prepare Excel for download (Processed sheet)
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Processed")
            output.seek(0)

            st.download_button(
                label="Download Processed.xlsx",
                data=output,
                file_name="Processed_FIFO_by_order.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("No files uploaded yet. Select Excel/CSV files to begin.")
