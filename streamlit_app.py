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

def normalize_column_names(df):
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def find_fmv_for_row(row):
    for c in FMV_CANDIDATES:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except:
                pass
    return None


# ---------------- FIFO ENGINE ----------------

def process_merged_dataframe(df, apply_grandfather=False):
    df = df.copy()
    df = normalize_column_names(df)

    # detect qty column
    qty_candidates = [COL_QTY, "Qty", "Quantity", "Quantity Sold"]
    qty_col = next((c for c in qty_candidates if c in df.columns), COL_QTY)

    # ensure required columns exist
    for c in (COL_ISIN, COL_TYPE, COL_SALE_DATE, COL_PUR_DATE,
              qty_col, COL_SALE_PRICE, COL_PUR_PRICE):
        if c not in df.columns:
            df[c] = pd.NA

    # parsing
    df[COL_TYPE] = df[COL_TYPE].astype(str).fillna("").str.strip()
    df["Type_norm"] = df[COL_TYPE].str.upper()

    df[COL_PUR_DATE] = pd.to_datetime(df[COL_PUR_DATE], errors="coerce")
    df[COL_SALE_DATE] = pd.to_datetime(df[COL_SALE_DATE], errors="coerce")

    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)
    df[COL_SALE_PRICE] = pd.to_numeric(df[COL_SALE_PRICE], errors="coerce")
    df[COL_PUR_PRICE] = pd.to_numeric(df[COL_PUR_PRICE], errors="coerce")

    df["Holding Days"] = (df[COL_SALE_DATE] - df[COL_PUR_DATE]).dt.days

    # outputs
    df[COL_SHORT] = 0.0
    df[COL_LONG] = 0.0
    df[COL_CAP_GAIN] = 0.0
    df[COL_FIFO_SELL_ORDER] = pd.NA
    df[COL_ISIN_COMPLETE] = pd.NA

    # capital gain = (sale - pur) * qty
    capgains = []
    for _, row in df.iterrows():
        q = float(row[qty_col] or 0)
        sp = row[COL_SALE_PRICE]
        pp = row[COL_PUR_PRICE]
        if q and pd.notna(sp) and pd.notna(pp):
            capgains.append(round((float(sp) - float(pp)) * q, 2))
        else:
            capgains.append(0.0)
    df[COL_CAP_GAIN] = capgains

    # ---------- FIFO per ISIN ----------
    global_sell_counter = 1
    isin_completion = {}

    for isin, group in df.groupby(COL_ISIN, sort=False):
        g = group.copy().reset_index()
        buy_queue = []
        bought = 0.0
        sold = 0.0
        completed = False

        for _, row in g.iterrows():
            idx = int(row["index"])
            qty = float(row[qty_col] or 0.0)
            rtype = row["Type_norm"]
            sale_price = row[COL_SALE_PRICE]
            buy_price = row[COL_PUR_PRICE]
            sale_date = row[COL_SALE_DATE]
            buy_date = row[COL_PUR_DATE]
            row_fmv = find_fmv_for_row(row)

            if rtype.startswith("B"):     # BUY
                if qty > 0:
                    buy_queue.append({
                        "qty": qty,
                        "orig_buy_price": float(buy_price) if pd.notna(buy_price) else 0.0,
                        "buy_date": buy_date
                    })
                    bought += qty

            elif rtype.startswith("S"):   # SELL
                remaining = qty

                # consume FIFO buys
                while remaining > 0 and buy_queue:
                    lot = buy_queue[0]
                    take = min(remaining, lot["qty"])

                    adj_cost = lot["orig_buy_price"]
                    if apply_grandfather and row_fmv is not None:
                        adj_cost = max(adj_cost, row_fmv)

                    alloc = (float(sale_price) - adj_cost) * take

                    # ST/LT
                    hd = None
                    if pd.notna(lot["buy_date"]) and pd.notna(sale_date):
                        hd = int((sale_date - lot["buy_date"]).days)

                    if hd and hd > LONG_TERM_DAYS:
                        df.loc[idx, COL_LONG] += round(alloc, 2)
                    else:
                        df.loc[idx, COL_SHORT] += round(alloc, 2)

                    lot["qty"] -= take
                    remaining -= take
                    sold += take

                    if lot["qty"] <= 0:
                        buy_queue.pop(0)

                # unmatched qty -> fallback to row's purchase price
                if remaining > 0:
                    adj = float(buy_price) if pd.notna(buy_price) else 0.0
                    if apply_grandfather and row_fmv is not None:
                        adj = max(adj, row_fmv)

                    alloc = (float(sale_price) - adj) * remaining

                    hd = None
                    if pd.notna(buy_date) and pd.notna(sale_date):
                        hd = int((sale_date - buy_date).days)

                    if hd and hd > LONG_TERM_DAYS:
                        df.loc[idx, COL_LONG] += round(alloc, 2)
                    else:
                        df.loc[idx, COL_SHORT] += round(alloc, 2)

                    sold += remaining
                    remaining = 0

                # assign global FIFO sell index
                df.loc[idx, COL_FIFO_SELL_ORDER] = global_sell_counter
                this_sell = global_sell_counter
                global_sell_counter += 1

                # completion check
                if not completed and bought > 0 and sold >= bought:
                    isin_completion[isin] = this_sell
                    completed = True

        df.loc[df[COL_ISIN] == isin, COL_ISIN_COMPLETE] = isin_completion.get(isin, pd.NA)

    # final date formatting
    df[COL_PUR_DATE] = pd.to_datetime(df[COL_PUR_DATE], errors="coerce").dt.strftime("%d/%m/%Y")
    df[COL_SALE_DATE] = pd.to_datetime(df[COL_SALE_DATE], errors="coerce").dt.strftime("%d/%m/%Y")

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
            df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
            dfs.append(normalize_column_names(df))

        merged = pd.concat(dfs, ignore_index=True)

        st.subheader("Merged (Original Row Order)")
        st.dataframe(merged.head(100))

        if st.button("Process & Download"):
            result = process_merged_dataframe(merged, apply_grandfather)

            # ⭐ FINAL SORTING RULE ⭐
            # 1. Keep ISIN groups together
            # 2. Sort inside each ISIN by SALE DATE ascending
            result["_orig"] = result.index
            result[COL_SALE_DATE] = pd.to_datetime(result[COL_SALE_DATE], errors="coerce")

            result = result.sort_values(
                by=[COL_ISIN, COL_SALE_DATE, "_orig"],
                na_position="last"
            ).reset_index(drop=True)

            result = result.drop(columns=["_orig"], errors="ignore")

            st.subheader("Final Output (Assets Together, Sale-Date ASC)")
            st.dataframe(result.head(200))

            # download file
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
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
