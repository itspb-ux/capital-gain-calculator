# streamlit_app.py
import streamlit as st
import pandas as pd
from io import BytesIO
import re

# ----- Helpers -----
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns by:
      - replacing any whitespace/newlines with a single space
      - stripping leading/trailing spaces
    This turns 'FMV Price on\\n31-Jan-2018' into 'FMV Price on 31-Jan-2018'.
    """
    df.columns = [
        re.sub(r"\s+", " ", str(col)).strip()
        for col in df.columns
    ]
    return df


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

# FMV header candidates (after normalization)
FMV_CANDIDATES = [
    "FMV",
    "FMV_31Jan2018",
    "FMV_31_01_2018",
    "FMV_31-01-2018",
    "FMV on 31 Jan 2018",
    "FMV Price on 31-Jan-2018",   # your header becomes this after normalization
    "FMV Price on 31 Jan 2018",
]


def build_fmv_map_from_df(
    df_map: pd.DataFrame,
    isin_col_candidates=None,
    fmv_col_candidates=None,
):
    """
    Build a dict ISIN -> FMV from a mapping sheet.
    """
    if isin_col_candidates is None:
        isin_col_candidates = [COL_ISIN, "ISIN", "isin", "Ticker", "Symbol"]
    if fmv_col_candidates is None:
        fmv_col_candidates = FMV_CANDIDATES

    isin_col = None
    fmv_col = None

    for c in isin_col_candidates:
        if c in df_map.columns:
            isin_col = c
            break

    for c in fmv_col_candidates:
        if c in df_map.columns:
            fmv_col = c
            break

    if isin_col is None or fmv_col is None:
        return {}

    tmp = df_map[[isin_col, fmv_col]].dropna()
    tmp[fmv_col] = pd.to_numeric(tmp[fmv_col], errors="coerce")
    tmp = tmp.dropna(subset=[fmv_col])

    mapping = dict(
        zip(
            tmp[isin_col].astype(str).str.strip(),
            tmp[fmv_col].astype(float),
        )
    )
    return mapping


def find_fmv_for_row(row, fmv_map, fmv_col_candidates=FMV_CANDIDATES):
    """
    Return FMV for a row:
      1) Prefer any FMV column present in this row.
      2) Else, lookup by ISIN in fmv_map.
      3) Else, None.
    """
    # 1) direct FMV column on row
    for c in fmv_col_candidates:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                continue

    # 2) mapping by ISIN
    isin = str(row.get(COL_ISIN, "")).strip()
    if isin and isin in fmv_map:
        return float(fmv_map[isin])

    return None


def process_merged_dataframe(
    df: pd.DataFrame,
    apply_grandfather: bool = False,
    fmv_map=None,
) -> pd.DataFrame:
    """
    Processor with optional Grandfathering (31-Jan-2018).

    - If apply_grandfather=True and FMV data is available, cost basis is:
          cost = min( max(actual_buy_price, FMV_31Jan2018), sale_price )

    - Supports:
        * FIFO mode when there is a 'Type' column (BUY / SELL).
        * Simple per-row mode when there is no 'Type' column.
    """
    df = df.copy()
    df = normalize_column_names(df)
    df.columns = [str(c).strip() for c in df.columns]

    # quantity column detection
    qty_col_candidates = [COL_QTY, "Qty", "Quantity", "Quantity Sold"]
    for c in qty_col_candidates:
        if c in df.columns:
            qty_col = c
            break
    else:
        qty_col = COL_QTY  # default; may be empty

    # ensure necessary columns exist
    for col in (COL_SALE_DATE, COL_PUR_DATE, qty_col, COL_SALE_PRICE, COL_PUR_PRICE):
        if col not in df.columns:
            df[col] = pd.NA

    # parse core columns
    df[COL_SALE_DATE] = pd.to_datetime(df.get(COL_SALE_DATE), errors="coerce")
    df[COL_PUR_DATE] = pd.to_datetime(df.get(COL_PUR_DATE), errors="coerce")
    df[qty_col] = pd.to_numeric(df.get(qty_col), errors="coerce").fillna(0.0)
    df[COL_SALE_PRICE] = pd.to_numeric(df.get(COL_SALE_PRICE), errors="coerce")
    df[COL_PUR_PRICE] = pd.to_numeric(df.get(COL_PUR_PRICE), errors="coerce")

    # output columns
    df[COL_SHORT] = 0.0
    df[COL_LONG] = 0.0
    df[COL_CAP_GAIN] = 0.0
    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    if "Type" in df.columns:
        df["Type_norm"] = df["Type"].astype(str).str.strip().str.upper()

    if fmv_map is None:
        fmv_map = {}

    # -------- FIFO MODE (if 'Type' column exists) --------
    if "Type" in df.columns:
        for isin, group in df.groupby(COL_ISIN, sort=False):
            g = group.copy().reset_index()  # 'index' is original df index
            g["_tx_date"] = g[COL_SALE_DATE].fillna(g[COL_PUR_DATE])
            g = g.sort_values("_tx_date").reset_index(drop=True)

            buy_queue = []  # list of dicts: {qty, buy_price, buy_date}
            st_map = {}
            lt_map = {}
            tot_map = {}

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
                tot_map[orig_idx] = 0.0

                row_fmv = find_fmv_for_row(r, fmv_map)

                if rtype.startswith("B"):  # BUY
                    if qty > 0:
                        lot_cost = float(buy_price) if pd.notna(buy_price) else 0.0
                        if apply_grandfather and row_fmv is not None:
                            lot_cost = max(lot_cost, row_fmv)
                        buy_queue.append(
                            {
                                "qty": qty,
                                "buy_price": lot_cost,
                                "buy_date": buy_date if pd.notna(buy_date) else pd.NaT,
                            }
                        )

                elif rtype.startswith("S"):  # SELL
                    qty_to_sell = qty
                    if qty_to_sell <= 0 or pd.isna(sale_price) or pd.isna(sale_date):
                        continue

                    # consume buy lots FIFO
                    while qty_to_sell > 0 and buy_queue:
                        lot = buy_queue[0]
                        take_qty = min(qty_to_sell, lot["qty"])
                        buy_p = lot["buy_price"]
                        gain = (sale_price - buy_p) * take_qty

                        hd = None
                        if pd.notna(lot["buy_date"]) and pd.notna(sale_date):
                            try:
                                hd = int((sale_date - lot["buy_date"]).days)
                            except Exception:
                                hd = None

                        if hd is not None and hd > LONG_TERM_DAYS:
                            lt_map[orig_idx] += gain
                        else:
                            st_map[orig_idx] += gain
                        tot_map[orig_idx] += gain

                        lot["qty"] -= take_qty
                        qty_to_sell -= take_qty
                        if lot["qty"] <= 0:
                            buy_queue.pop(0)

                    # unmatched SELL quantity (no or insufficient buys)
                    if qty_to_sell > 0:
                        cost_basis = float(buy_price) if pd.notna(buy_price) else 0.0
                        if apply_grandfather and row_fmv is not None:
                            cost_basis = max(cost_basis, row_fmv)
                        gain = (sale_price - cost_basis) * qty_to_sell

                        hd = None
                        if pd.notna(buy_date) and pd.notna(sale_date):
                            try:
                                hd = int((sale_date - buy_date).days)
                            except Exception:
                                hd = None

                        if hd is not None and hd > LONG_TERM_DAYS:
                            lt_map[orig_idx] += gain
                        else:
                            st_map[orig_idx] += gain
                        tot_map[orig_idx] += gain

            # write back
            for idx_key, v in st_map.items():
                df.loc[idx_key, COL_SHORT] = round(v, 2)
            for idx_key, v in lt_map.items():
                df.loc[idx_key, COL_LONG] = round(v, 2)
            for idx_key, v in tot_map.items():
                df.loc[idx_key, COL_CAP_GAIN] = round(v, 2)

        df = df.drop(columns=["Type_norm", "_tx_date"], errors="ignore")

    # -------- SIMPLE PER-ROW MODE (no 'Type' column) --------
    else:
        short_list = []
        long_list = []
        total_list = []

        for _, row in df.iterrows():
            q = float(row.get(qty_col) or 0.0)
            sp = row.get(COL_SALE_PRICE)
            pp = row.get(COL_PUR_PRICE)
            hd = row.get("Holding Days")

            row_fmv = find_fmv_for_row(row, fmv_map)

            if q == 0 or pd.isna(sp) or pd.isna(pp):
                short_list.append(0.0)
                long_list.append(0.0)
                total_list.append(0.0)
                continue

            cost_basis = float(pp)
            if apply_grandfather and row_fmv is not None:
                cost_basis = max(cost_basis, row_fmv)

            # cost basis cannot exceed sale price
            try:
                cost_basis = min(cost_basis, float(sp))
            except Exception:
                pass

            gain = (float(sp) - cost_basis) * q

            if pd.notna(hd) and hd > LONG_TERM_DAYS:
                st_val, lt_val = 0.0, gain
            else:
                st_val, lt_val = gain, 0.0

            short_list.append(round(st_val, 2))
            long_list.append(round(lt_val, 2))
            total_list.append(round(gain, 2))

        df[COL_SHORT] = short_list
        df[COL_LONG] = long_list
        df[COL_CAP_GAIN] = total_list

    # final date formatting
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
    "They will be merged and Short/Long term capital gains will be calculated."
)

# --- Grandfathering section ---
st.markdown("### Grandfathering (31-Jan-2018)")
apply_grandfather = st.checkbox(
    "Apply Grandfathering using FMV as on 31-Jan-2018"
)

fmv_map = {}
if apply_grandfather:
    st.info(
        "The app will look for an FMV column in your data "
        "(for example 'FMV Price on 31-Jan-2018'). "
        "You can also upload a separate mapping file (ISIN → FMV)."
    )
    fmv_file = st.file_uploader(
        "Optional: Upload ISIN → FMV mapping file (Excel/CSV)",
        type=["xlsx", "xls", "csv"],
        key="fmv_file",
    )
    if fmv_file is not None:
        try:
            if fmv_file.name.lower().endswith(".csv"):
                map_df = pd.read_csv(fmv_file)
            else:
                map_df = pd.read_excel(fmv_file, header=0)
            map_df = normalize_column_names(map_df)
            fmv_map = build_fmv_map_from_df(map_df)
            if not fmv_map:
                st.warning(
                    "Could not detect ISIN and FMV columns in the mapping file. "
                    "Make sure it has 'ISIN' and an FMV column."
                )
            else:
                st.success(
                    f"Loaded FMV mapping for {len(fmv_map)} ISIN(s)."
                )
        except Exception as e:
            st.error(f"Error reading FMV mapping file: {e}")

# --- Main file uploader ---
uploaded = st.file_uploader(
    "Select Excel/CSV files to merge & process",
    accept_multiple_files=True,
    type=["xlsx", "xls", "csv"],
    key="data_files",
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
        st.subheader("Merged data preview")
        st.dataframe(merged.head(50))

        if st.button("Process & Download Excel"):
            merged_fmv_cols = [c for c in merged.columns if c in FMV_CANDIDATES]
            merged_has_fmv = len(merged_fmv_cols) > 0

            if apply_grandfather and (not merged_has_fmv) and (not fmv_map):
                st.warning(
                    "Grandfathering is selected but no FMV column was found in "
                    "the uploaded data and no mapping file was provided. "
                    "Processing will continue **without** applying FMV."
                )

            result_df = process_merged_dataframe(
                merged,
                apply_grandfather=apply_grandfather,
                fmv_map=fmv_map,
            )

            st.subheader("Processed preview")
            st.dataframe(result_df.head(50))

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Processed")
            output.seek(0)

            st.download_button(
                label="Download Merged_with_gains.xlsx",
                data=output,
                file_name="Merged_with_gains.xlsx",
                mime=(
                    "application/"
                    "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
            )
    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info(
        "No files uploaded yet. Select one or more Excel/CSV files from your computer."
    )
