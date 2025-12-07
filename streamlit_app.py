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
    Process merged dataframe:
      - FIFO matching per ISIN: buys consumed by purchase-date FIFO
      - Sells processed by sale-date ascending within each ISIN
      - Output groups rows by ISIN and preserves FIFO_Sell_Order for arrange-by-sell-order
    """
    df = df.copy()
    df = normalize_column_names(df)

    # detect qty column
    qty_col_candidates = [COL_QTY, "Qty", "Quantity", "Quantity Sold"]
    qty_col = next((c for c in qty_col_candidates if c in df.columns), COL_QTY)

    # ensure required columns exist
    for col in (COL_ISIN, COL_SALE_DATE, COL_PUR_DATE, qty_col, COL_SALE_PRICE, COL_PUR_PRICE):
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

    # normalize type if present
    has_type = "Type" in df.columns
    if has_type:
        df["Type_norm"] = df["Type"].astype(str).str.strip().str.upper()

    # global sell counter to assign FIFO order across all ISINs
    global_sell_counter = 1

    # ---------- FIFO per ISIN, sells ordered by sale date ascending ----------
    if has_type:
        # group by ISIN but preserve group order from sorted merged df
        for isin, group in df.groupby(COL_ISIN, sort=False):
            g = group.copy().reset_index()  # keep original index in 'index'

            # separate buys and sells
            buys = g[g["Type_norm"].str.startswith("B", na=False)].copy()
            sells = g[g["Type_norm"].str.startswith("S", na=False)].copy()

            # For buys: sort by Pur. Date ascending (missing Pur. Date go last)
            buys["_pur_sort"] = buys[COL_PUR_DATE].fillna(pd.Timestamp("9999-12-31"))
            buys = buys.sort_values("_pur_sort").reset_index(drop=True)

            # For sells: sort by Sale Date ascending (missing Sale Date go last)
            sells["_sale_sort"] = sells[COL_SALE_DATE].fillna(pd.Timestamp("9999-12-31"))
            sells = sells.sort_values("_sale_sort").reset_index(drop=True)

            # buy queue (lots eligible to be consumed)
            buy_queue = []
            buy_idx = 0

            # maps for accumulating ST/LT per sell original index
            st_map = {}
            lt_map = {}

            for _, srow in sells.iterrows():
                st_map[int(srow["index"])] = 0.0
                lt_map[int(srow["index"])] = 0.0

            # process sells in sale-date order
            for _, srow in sells.iterrows():
                sell_idx = int(srow["index"])
                sell_qty = float(srow.get(qty_col) or 0.0)
                sell_price = srow.get(COL_SALE_PRICE)
                sell_date = srow.get(COL_SALE_DATE)
                if sell_qty <= 0 or pd.isna(sell_price) or pd.isna(sell_date):
                    continue

                # add buys whose Pur. Date <= this sell's Sale Date (eligible buys)
                while buy_idx < len(buys):
                    cand = buys.loc[buy_idx]
                    cand_pur = cand[COL_PUR_DATE]
                    # missing pur date -> treat as unavailable for earlier sells (they were put last)
                    if pd.isna(cand_pur):
                        break
                    if cand_pur <= sell_date:
                        lot_qty = float(cand.get(qty_col) or 0.0)
                        orig_buy_price = float(cand.get(COL_PUR_PRICE) or 0.0)
                        lot_buy_date = cand.get(COL_PUR_DATE)
                        if lot_qty > 0:
                            buy_queue.append({"qty": lot_qty, "orig_buy_price": orig_buy_price, "buy_date": lot_buy_date})
                        buy_idx += 1
                    else:
                        break

                qty_to_sell = sell_qty

                # consume from buy_queue FIFO
                while qty_to_sell > 0 and len(buy_queue) > 0:
                    lot = buy_queue[0]
                    take = min(qty_to_sell, lot["qty"])

                    adjusted_cost = lot["orig_buy_price"]
                    row_fmv = find_fmv_for_row(srow)
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)
                    # do NOT cap adjusted_cost at sale price (allow negative allocations)

                    alloc = (float(sell_price) - adjusted_cost) * take

                    # compute holding days
                    hd = None
                    if pd.notna(lot["buy_date"]) and pd.notna(sell_date):
                        try:
                            hd = int((sell_date - lot["buy_date"]).days)
                        except Exception:
                            hd = None

                    if hd is not None and hd > LONG_TERM_DAYS:
                        lt_map[sell_idx] += alloc
                    else:
                        st_map[sell_idx] += alloc

                    # update quantities
                    lot["qty"] -= take
                    qty_to_sell -= take

                    if lot["qty"] <= 0:
                        buy_queue.pop(0)
                    else:
                        buy_queue[0] = lot

                # remaining unmatched qty -> fallback to sell-row pur price if present
                if qty_to_sell > 0:
                    fallback_cost = float(srow.get(COL_PUR_PRICE) or 0.0)
                    adjusted_cost = fallback_cost
                    row_fmv = find_fmv_for_row(srow)
                    if apply_grandfather and row_fmv is not None:
                        adjusted_cost = max(adjusted_cost, row_fmv)

                    alloc = (float(sell_price) - adjusted_cost) * qty_to_sell

                    hd = None
                    sell_row_buy_date = srow.get(COL_PUR_DATE)
                    if pd.notna(sell_row_buy_date) and pd.notna(sell_date):
                        try:
                            hd = int((sell_date - sell_row_buy_date).days)
                        except Exception:
                            hd = None

                    if hd is not None and hd > LONG_TERM_DAYS:
                        lt_map[sell_idx] += alloc
                    else:
                        st_map[sell_idx] += alloc

                    qty_to_sell = 0.0

                # assign global FIFO sell order for this sell
                df.loc[sell_idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                global_sell_counter += 1

            # write back ST/LT into df using original indices
            for idx_key, v in st_map.items():
                df.loc[idx_key, COL_SHORT] = round(v, 2)
            for idx_key, v in lt_map.items():
                df.loc[idx_key, COL_LONG] = round(v, 2)

        # cleanup helper cols if any
        df = df.drop(columns=["Type_norm", "_pur_sort", "_sale_sort"], errors="ignore")

    else:
        # simple per-row mode: allocate by row's holding days
        short_list = []
        long_list = []
        for _, row in df.iterrows():
            q = float(row.get(qty_col) or 0.0)
            sp = row.get(COL_SALE_PRICE)
            pp = row.get(COL_PUR_PRICE)
            hd = row.get("Holding Days")
            if q == 0 or pd.isna(sp) or pd.isna(pp):
                short_list.append(0.0); long_list.append(0.0); continue
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
st.set_page_config(page_title="Capital Gain Calculator (grouped by ISIN, sale-date FIFO)", layout="centered")

st.title("Capital Gain Calculator")
st.write(
    "Upload Excel/CSV files. The app will merge them, group rows by ISIN, sort sells by Sale Date ascending within each ISIN, "
    "and apply FIFO (purchase-date) for allocation. Optional FMV (31-Jan-2018) must be present in uploaded files to apply grandfathering."
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

        merged = pd.concat(dfs, ignore_index=True)

        # --- IMPORTANT ORDERING: group by ISIN and within group sort by Sale Date ascending ---
        # parse dates to sort
        merged[COL_SALE_DATE] = pd.to_datetime(merged.get(COL_SALE_DATE), errors="coerce")
        merged[COL_PUR_DATE] = pd.to_datetime(merged.get(COL_PUR_DATE), errors="coerce")

        # sort by ISIN ascending then Sale Date ascending (missing Sale Date placed last), then Pur. Date ascending
        merged = merged.sort_values(by=[COL_ISIN, COL_SALE_DATE, COL_PUR_DATE], na_position="last").reset_index(drop=True)

        st.subheader("Merged (grouped by ISIN; sells sorted by Sale Date asc) preview")
        st.dataframe(merged.head(120))

        if st.button("Process & Download Excel"):
            # check FMV presence
            merged_fmv_cols = [c for c in merged.columns if c in FMV_CANDIDATES]
            if apply_grandfather and (not merged_fmv_cols):
                st.warning("Grandfathering selected but no FMV column found; processing will continue without FMV.")

            result_df = process_merged_dataframe(merged, apply_grandfather=apply_grandfather)

            st.subheader("Processed preview (grouped by ISIN; sells sorted by FIFO_Sell_Order)")
            st.dataframe(result_df.head(200))

            # show sells arranged by FIFO_Sell_Order first (so you can inspect matched sells in order)
            sells_sorted = result_df[result_df[COL_FIFO_SELL_ORDER].notna()].sort_values(by=[COL_ISIN, COL_FIFO_SELL_ORDER])
            others = result_df[result_df[COL_FIFO_SELL_ORDER].isna()]
            arranged = pd.concat([sells_sorted, others], ignore_index=True)

            st.markdown("**Sells (by ISIN, FIFO sell order)**")
            st.dataframe(sells_sorted.head(200))

            # prepare Excel with two sheets: full processed and sells-first arranged
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Processed")
                arranged.to_excel(writer, index=False, sheet_name="Sells_First_By_FIFO")
            output.seek(0)

            st.download_button(
                label="Download Processed.xlsx",
                data=output,
                file_name="Processed_with_FIFO.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("No files uploaded yet. Select Excel/CSV files to begin.")
