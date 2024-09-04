import os
import glob
import argparse
import pandas as pd


def parse_time(time_str):
    time_str = time_str.split("(")[0].strip()
    ts = pd.to_datetime(time_str)
    return ts


def parse_time_row(row, col):
    time_str = row[col]
    return parse_time(time_str)


def detect_gap(fn=None, threshold=2, its_col='ITS_File_Name'):
    '''
    params:
        fn: dataframe filepath or dataframe
        threshold: > threshold seconds gap in recording is called gap in recording
    return: dataframe recording gaps in file
    '''
    if isinstance(fn, str):
        if fn.endswith('.csv'):
            df = pd.read_csv(fn)
        elif fn.endswith('.xlsx'):
            df = pd.read_excel(fn, engine='openpyxl')
        else:
            raise NotImplementedError(f"The file type of {fn} must be .csv or .xlsx")
    else:
        df = fn
        fn = 'Unknown'

    records = []
    grped = df.groupby(its_col)
    for its, grp in grped:
        gap_num = 0
        grp['StartTime_ts'] = grp.apply(parse_time_row, args=('StartTime', ), axis=1)
        grp['EndTime_ts'] = grp.apply(parse_time_row, args=('EndTime', ), axis=1)
        for i, row in grp.reset_index().iterrows():
            this_end = row['EndTime_ts']
            if (i+1) < len(grp):
                next_start = grp.iloc[i+1]['StartTime_ts']
                gap = (next_start - this_end).seconds
                if gap > threshold:
                    record = {
                        'Filename': os.path.basename(fn),
                        'ItsFile': its,
                        'NthGap': gap_num,
                        'Index': row['index'],
                        'GapTime_seconds': gap,
                        'EndTime': this_end,
                        'NextStartTime': next_start,
                        'Filepath': os.path.abspath(fn)
                    }
                    gap_num += 1
                    records.append(record)

    return pd.DataFrame(records), len(grped.groups.keys())


def detect_gap_wrapper(fns, threshold=2, its_col='ITS_File_Name'):
    '''
    params:
        threshold: > threshold seconds gap in recording is called gap in recording
    return: dataframe recording gaps in file
    '''
    summary = []
    dfs = []
    for fn in fns:
        print(f'Detecting gaps in {fn}')
        df, n_its = detect_gap(fn, threshold, its_col)
        if len(df):
            r = {
                'Filename': os.path.basename(fn),
                'N_ItsFile': n_its,
                'N_ItsFile_with_gap': len(df['ItsFile'].value_counts()),
                'ItsFile_with_gap': ",".join(list(df['ItsFile'].unique())),
                'Filepath': os.path.abspath(fn)
                
            }
        else:
            r = {
                'Filename': os.path.basename(fn),
                'N_ItsFile': n_its,
                'N_ItsFile_with_gap': 0,
                'ItsFile_with_gap': "",
                'Filepath': os.path.abspath(fn)
            }
        summary.append(r)
        dfs.append(df)
    df = pd.concat(dfs)
    summary = pd.DataFrame(summary)
    n_gap = len(summary[summary['N_ItsFile_with_gap'] > 0])
    print(f'{n_gap} out of {len(summary)} files has time gaps in itsfile')
    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Detect gaps in itsfiles")
    parser.add_argument('-f', '--files', required=True, nargs='+',
                        help="input LENAExport 5Minute csv files; wildcard * pattern matching allowed.")
    parser.add_argument('-o', '--output', default='LENAExport_ItsFile_gap_report',
                        help="Output prefix [Default: %(default)s]")
    args = parser.parse_args()

    fns = []
    for fn in args.files:
        fns.extend(glob.glob(fn))
    df, summary = detect_gap_wrapper(fns, threshold=1, its_col='ITS_File_Name')
    outfn_1 = f"{args.output}_details.csv"
    outfn_2 = f"{args.output}_summary.csv"
    df.to_csv(outfn_1)
    summary.to_csv(outfn_2)
    print(f'Write output to:\n{outfn_1}\n{outfn_2}')

if __name__ == '__main__':
    main()