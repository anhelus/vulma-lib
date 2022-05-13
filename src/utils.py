import pandas as pd


def to_csv(history, outfile):
    hist_df = pd.DataFrame(history)
    with open(outfile, mode='w') as f:
        hist_df.to_csv(f)
