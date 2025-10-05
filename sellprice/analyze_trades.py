import pandas as pd
from datetime import datetime, timezone
from tabulate import tabulate
from utils import print_signal_summary


def run_post_analysis():
    from tabulate import tabulate
    import pandas as pd

    try:
        signals = print_signal_summary()  # your existing function that returns list of dicts
        if not signals:
            print("[INFO] No signals to summarize.")
            return

        df = pd.DataFrame(signals)

        # Confidence floors by type (keeps the blotter tight)
        floors = {'SHORT': 55, 'SELL': 50, 'BUY': 60}
        df = df[df.apply(lambda r: r['Confidence'] >= floors.get(r['Type'], 50), axis=1)].copy()

        # Sort & persist
        df.sort_values(['Type','Confidence'], ascending=[True, False], inplace=True)
        df.to_csv("signal_summary.csv", index=False)
        print("💾 Saved: signal_summary.csv")

        # Display
        _print_top_by_stage(df, top_n=5)
        print("\n📊 Trade Signal Summary Table:\n")
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))

        # Optional: show Top-N per Type for quicker scanning
        TOP_N = 5
        buckets = []
        for t in ['SHORT', 'SELL', 'BUY']:
            sub = df[df['Type'] == t].nlargest(TOP_N, 'Confidence')
            if not sub.empty:
                print(f"\nTop {min(TOP_N, len(sub))} {t} signals:")
                print(tabulate(sub, headers='keys', tablefmt='fancy_grid', showindex=False))
                buckets.append(sub)
        # If you want the overall table too, keep it as-is below


    except Exception as e:
        print(f"⚠️ Error during post-analysis summary: {e}")



def _norm_type(t: str) -> str:
    t = (t or "").upper()
    return {"BUY","SELL","SHORT","WATCH"}.intersection({t}).pop() if t in {"BUY","SELL","SHORT","WATCH"} else "WATCH"



def _print_top_by_stage(df_all, top_n=5):
    # Keep only the Stage rows we logged from sell.py
    stage_rows = df_all[(df_all['Signal'].str.startswith('Stage:')) & (df_all['Type'].str.upper() == 'WATCH')].copy()
    if stage_rows.empty:
        print("\n📭 No stage rows found.")
        return

    # Extract stage label
    stage_rows['Stage'] = stage_rows['Signal'].str.replace('Stage:', '').str.strip()

    def _bucket(label):
        if label.startswith('Mark-Up'): return 'Mark-Up'
        if label.startswith('Accumulation'): return 'Accumulation'
        if label.startswith('Distribution'): return 'Distribution'
        if label.startswith('Mark-Down'): return 'Mark-Down'
        return 'Neutral/Transition'

    stage_rows['Bucket'] = stage_rows['Stage'].apply(_bucket)

    for bucket in ['Mark-Up', 'Accumulation', 'Distribution', 'Mark-Down', 'Neutral/Transition']:
        sub = stage_rows[stage_rows['Bucket'] == bucket].nlargest(top_n, 'Confidence')
        if sub.empty:
            continue
        print(f"\nTop {min(top_n, len(sub))} {bucket}:")
        print(tabulate(sub[['Symbol','Date','Stage','Confidence']], headers='keys', tablefmt='fancy_grid', showindex=False))

# Call this at the end of run_post_analysis(), after the existing prints:
#   _print_top_by_stage(df)
