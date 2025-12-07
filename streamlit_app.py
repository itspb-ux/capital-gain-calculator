# streamlit_app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import re
import math

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
    - Start with merged dataframe sorted by Pur. Date ascending (the UI sorts before this call).
    - For each ISIN, process rows in the order they appear in the merged (purchase-date) sequence.
    - Buys are queued FIFO in that order; sells consume that queue FIFO.
    - Track global FIFO_Sell_Order and for each ISIN compute the ISIN_Completion_Order
      (the global sell order where cumulative sold qty >= cumulative bought qty for that ISIN).
    - Final output is augmented with FIFO_Sell_Order and ISIN_Completion_Order.
    - Final sorting (outside) will be by earliest Pur. Date per ISIN then ISIN_Completion_Order.
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

    # parse types/dates/numerics
    df[COL_TYPE] = df.get(COL_TYPE).astype(str).fillna("").str.strip()
    df["Type_norm"] = df[COL_TYPE].str.upper()
    df[COL_SALE_DATE] = pd.to_datetime(df.get(COL_SALE_DATE), errors="coerce")
    df[COL_PUR_DATE] = pd.to_datetime(df.get(COL_PUR_DATE), errors="coerce")
    df[qty_col] = pd.to_numeric(df.get(qty_col), errors="coerce").fillna(0.0)
    df[COL_SALE_PRICE] = pd.to_numeric(df.get(COL_SALE_PRICE), errors="coerce")
    df[COL_PUR_PRICE] = pd.to_numeric(df.get(COL_PUR_PRICE), errors="coerce")

    # helper: holding days
    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    # initialize output columns
    df[COL_SHORT] = 0.0
    df[COL_LONG] = 0.0
    df[COL_CAP_GAIN] = 0.0
    df[COL_FIFO_SELL_ORDER] = pd.NA
    df[COL_ISIN_COMPLETE] = pd.NA

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

    # We'll process each ISIN in the order they appear in the merged df.
    # Maintain a global sell counter to assign FIFO sell order
    global_sell_counter = 1

    # track per-ISIN completion: when cumulative sold qty >= cumulative bought qty
    isin_completion_map = {}  # isin -> completion_order (global_sell_counter when completed)
    isin_bought_qty = {}      # total bought qty encountered (so far)
    isin_sold_qty = {}        # total sold qty encountered (so far)

    # group by ISIN but preserve merged order (groupby with sort=False does that)
    for isin, group in df.groupby(COL_ISIN, sort=False):
        g = group.copy().reset_index()  # keep original df index in 'index'
        # buy queue: list of dicts {qty, orig_buy_price, buy_date}
        buy_queue = []

        # init per-ISIN counters
        isin_bought = 0.0
        isin_sold = 0.0
        completed_flag = False

        # iterate rows in merged (purchase-date) order for this ISIN
        for _, row in g.iterrows():
            orig_idx = int(row["index"])
            rtype = str(row["Type_norm"] or "").upper()
            qty = float(row.get(qty_col) or 0.0)
            sale_price = row.get(COL_SALE_PRICE)
            buy_price = row.get(COL_PUR_PRICE)
            sale_date = row.get(COL_SALE_DATE)
            buy_date = row.get(COL_PUR_DATE)

            # FMV from uploaded data only
            row_fmv = find_fmv_for_row(row)

            if rtype.startswith("B"):
                # BUY encountered: append to buy_queue
                if qty > 0:
                    orig_buy = float(buy_price) if pd.notna(buy_price) else 0.0
                    buy_queue.append({
                        "qty": qty,
                        "orig_buy_price": orig_buy,
                        "buy_date": buy_date
                    })
                    isin_bought += qty

            elif rtype.startswith("S"):
                # SELL encountered: consume from buy_queue FIFO
                qty_to_sell = qty
                if qty_to_sell <= 0 or pd.isna(sale_price):
                    # still count as a sell row but nothing to allocate
                    # assign FIFO sell order nonetheless
                    df.loc[orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                    global_sell_counter += 1
                    # update isin_sold (even if zero)
                    isin_sold += qty
                    # check completion
                    if (not completed_flag) and isin_bought > 0 and isin_sold >= isin_bought:
                        isin_completion_map[isin] = df.loc[orig_idx, COL_FIFO_SELL_ORDER]
                        completed_flag = True
                    continue

                # ensure we have counters in maps
                isin_sold += qty  # add total sold for completion tracking (we'll also increment as we consume)
                remaining_to_consume = qty_to_sell

                # consume FIFO buy_queue
                while remaining_to_consume > 0 and len(buy_queue) > 0:
                    lot = buy_queue[0]
                    take_qty = min(remaining_to_consume, lot["qty"])

                    adjusted_cost = lot["orig_buy_price"]
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)
                    # DO NOT cap adjusted_cost at sale price

                    alloc = (float(sale_price) - adjusted_cost) * take_qty

                    # holding days relative to lot
                    hd = None
                    if pd.notna(lot["buy_date"]) and pd.notna(sale_date):
                        try:
                            hd = int((sale_date - lot["buy_date"]).days)
                        except Exception:
                            hd = None

                    if hd is not None and hd > LONG_TERM_DAYS:
                        # long term
                        df.loc[orig_idx, COL_LONG] = round(df.loc[orig_idx, COL_LONG] + alloc, 2)
                    else:
                        # short term
                        df.loc[orig_idx, COL_SHORT] = round(df.loc[orig_idx, COL_SHORT] + alloc, 2)

                    # reduce lot and remaining qty
                    lot["qty"] -= take_qty
                    remaining_to_consume -= take_qty

                    if lot["qty"] <= 0:
                        buy_queue.pop(0)
                    else:
                        buy_queue[0] = lot

                # if remaining unmatched qty left, fallback to sell-row pur price
                if remaining_to_consume > 0:
                    fallback_cost = float(buy_price) if pd.notna(buy_price) else 0.0
                    adjusted_cost = fallback_cost
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)

                    alloc = (float(sale_price) - adjusted_cost) * remaining_to_consume

                    hd = None
                    if pd.notna(buy_date) and pd.notna(sale_date):
                        try:
                            hd = int((sale_date - buy_date).days)
                        except Exception:
                            hd = None

                    if hd is not None and hd > LONG_TERM_DAYS:
                        df.loc[orig_idx, COL_LONG] = round(df.loc[orig_idx, COL_LONG] + alloc, 2)
                    else:
                        df.loc[orig_idx, COL_SHORT] = round(df.loc[orig_idx, COL_SHORT] + alloc, 2)

                    remaining_to_consume = 0.0

                # assign FIFO sell order for this sell row
                df.loc[orig_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                sell_order_for_this_row = global_sell_counter
                global_sell_counter += 1

                # if asset completes here (cumulative sold >= cumulative bought) and not recorded yet, record completion order
                if (not completed_flag) and isin_bought > 0:
                    # Check current cumulative sold for ISIN (we used pre-increment isin_sold)
                    # To be safe, compute sold sum across processed sell rows up to current global_sell_counter:
                    # we kept isin_sold increment earlier; check if it's >= isin_bought
                    # But because we increased isin_sold before allocating, use that:
                    if isin_sold >= isin_bought:
                        isin_completion_map[isin] = sell_order_for_this_row
                        completed_flag = True

            else:
                # row has no Type: we leave ST/LT as 0 (handled by simple-mode earlier)
                pass

        # after finishing ISIN group, store completion if not set (None if not fully sold)
        if isin in isin_completion_map:
            df.loc[df[COL_ISIN] == isin, COL_ISIN_COMPLETE] = isin_completion_map[isin]
        else:
            df.loc[df[COL_ISIN] == isin, COL_ISIN_COMPLETE] = pd.NA

    # final: format dates for output
    for col in [COL_SALE_DATE, COL_PUR_DATE]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%d/%m/%Y")

    return df


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Capital Gain Calculator (PurDate + FIFO asset ordering)", layout="centered")

st.title("Capital Gain Calculator")
st.write(
    "Upload Excel/CSV files. The app will merge them, sort global rows by Purchase Date ascending, "
    "apply FIFO matching per asset using those buys, assign FIFO_Sell_Order and compute ISIN_Completion_Order. "
    "Final arrangement will be by earliest purchase date per ISIN, then by ISIN_Completion_Order."
)

# Grandfathering control
apply_grandfather = st.checkbox("Apply Grandfathering (use FMV as on 31-Jan-2018)", value=False)

# File uploader
uploaded = st.file_uploader(
    "Select trade files to merge & process (Excel/CSV). FMV must be in these files to apply grandfathering.",
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

        merged = pd.concat(dfs, ignore_index=True)

        # --- Sort merged globally by Purchase Date ascending (missing Pur. Date go last) ---
        merged[COL_PUR_DATE] = pd.to_datetime(merged.get(COL_PUR_DATE), errors="coerce")
        merged[COL_SALE_DATE] = pd.to_datetime(merged.get(COL_SALE_DATE), errors="coerce")
        merged = merged.sort_values(by=[COL_PUR_DATE], na_position="last").reset_index(drop=True)

        st.subheader("Merged (sorted by Purchase Date global) preview")
        st.dataframe(merged.head(120))

        if st.button("Process & Download Excel"):
            # check FMV presence if requested
            merged_fmv_cols = [c for c in merged.columns if c in FMV_CANDIDATES]
            if apply_grandfather and (not merged_fmv_cols):
                st.warning("Grandfathering selected but no FMV column found; processing will continue without FMV.")

            result_df = process_merged_dataframe(merged, apply_grandfather=apply_grandfather)

            # Compute per-ISIN earliest Pur. Date for sorting assets
            isin_min_pur = result_df.groupby(COL_ISIN)[COL_PUR_DATE].apply(
                lambda s: pd.to_datetime(s, errors="coerce").min()
            ).to_dict()

            # build a dataframe of ISIN ordering keys
            isin_order_df = pd.DataFrame({
                COL_ISIN: list(isin_min_pur.keys()),
                "min_pur_date": list(isin_min_pur.values())
            })

            # attach completion order (if present) from result_df (they are identical across rows of same ISIN)
            # take first non-null ISIN_Completion_Order per ISIN
            isin_completion = result_df.groupby(COL_ISIN)[COL_ISIN_COMPLETE].first().to_dict()
            isin_order_df[COL_ISIN_COMPLETE] = isin_order_df[COL_ISIN].map(isin_completion)

            # Replace NaT with far future so it sorts last
            isin_order_df["min_pur_date"] = pd.to_datetime(isin_order_df["min_pur_date"], errors="coerce")
            isin_order_df["min_pur_date"] = isin_order_df["min_pur_date"].fillna(pd.Timestamp("9999-12-31"))

            # For completion order NaN -> large number so incomplete assets sort after completed ones with same pur date
            isin_order_df[COL_ISIN_COMPLETE] = isin_order_df[COL_ISIN_COMPLETE].apply(
                lambda v: int(v) if (pd.notna(v) and v != "") else 10**9
            )

            # determine ISIN sort order by (min_pur_date asc, ISIN_Completion_Order asc)
            isin_order_df = isin_order_df.sort_values(by=["min_pur_date", COL_ISIN_COMPLETE]).reset_index(drop=True)
            isin_rank = {row[COL_ISIN]: i for i, row in isin_order_df.iterrows()}

            # create final arranged df sorted by ISIN rank then FIFO_Sell_Order within sells, else original order
            # We'll put sells (rows with FIFO_Sell_Order) first grouped by ISIN rank and FIFO order, then other rows
            sells = result_df[result_df[COL_FIFO_SELL_ORDER].notna()].copy()
            sells[COL_FIFO_SELL_ORDER] = sells[COL_FIFO_SELL_ORDER].astype(int)
            sells["isin_rank"] = sells[COL_ISIN].map(isin_rank)

            sells = sells.sort_values(by=["isin_rank", COL_FIFO_SELL_ORDER]).reset_index(drop=True)

            others = result_df[result_df[COL_FIFO_SELL_ORDER].isna()].copy()
            others["isin_rank"] = others[COL_ISIN].map(isin_rank)
            # place other rows grouped by isin_rank and preserve original order within group
            others = others.sort_values(by=["isin_rank"]).reset_index(drop=True)

            arranged = pd.concat([sells, others], ignore_index=True)

            st.subheader("Processed preview (arranged by asset Pur.Date then completion order)")
            st.dataframe(arranged.head(300))

            # prepare Excel: full processed + arranged
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Processed")
                arranged.to_excel(writer, index=False, sheet_name="Arranged_By_Asset")
            output.seek(0)

            st.download_button(
                label="Download Processed.xlsx",
                data=output,
                file_name="Processed_with_asset_ordering.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("No files uploaded yet. Select Excel/CSV files to begin.")
