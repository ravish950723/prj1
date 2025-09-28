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
