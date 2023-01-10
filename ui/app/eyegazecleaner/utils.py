import re
import traceback
import numpy as np
import pandas as pd
from datetime import timedelta
from app import app


def assign_trial_id(df, code_col=app.config["CODE_COL"], 
                    begin_code=app.config["BEGIN_CODE"],
                    trial_id_col=app.config["TRIAL_ID_COL"]):
    codes = df[code_col].values
    trial_ids = []
    trial_id = 0
    for c in codes:
        if c == begin_code:
            trial_id += 1
        trial_ids.append(trial_id)
    df[trial_id_col] = trial_ids
    return df


def convert_milisecond_to_frame(df, scale=0.03, file_type="raw_coding_file",
    onset_col=app.config["ONSET_COL"],
    offset_col=app.config["OFFSET_COL"]):
    ''' 30 frame in 1000 miliseconds; 1 miliseconds = 0.03 frame.

    Note that we are rounding the results not just taking the integer portion
    '''
    if file_type == "raw_coding_file":
        df[onset_col] = round(df[onset_col] * scale)
        df[offset_col] = round(df[offset_col] * scale)
        df[onset_col] = df[onset_col].astype(int)
        df[offset_col] = df[offset_col].astype(int)
    else:
        # trial summary file
        for col in df.columns:
            if col != app.config["TRIAL_ID_COL"]:
                df[col] = round(df[col] * scale)
                df[col] = df[col].astype(int)
    return df


def convert_frame_to_milisecond(df, scale=100/3, file_type="raw_coding_file",
    onset_col=app.config["ONSET_COL"],
    offset_col=app.config["OFFSET_COL"]):
    ''' 30 frame in 1000 miliseconds; 1 frame = 1000/30 miliseconds.

    Note that we are rounding the results not just taking the integer portion
    '''
    if file_type == "raw_coding_file":
        df[onset_col] = round(df[onset_col] * scale)
        df[offset_col] = round(df[offset_col] * scale)
        df[onset_col] = df[onset_col].astype(int)
        df[offset_col] = df[offset_col].astype(int)
    else:
        # trial summary file
        for col in df.columns:
            if col != app.config["TRIAL_ID_COL"]:
                df[col] = round(df[col] * scale)
                df[col] = df[col].astype(int)
    return df


def read_data(fn, filepath, original_timestamp_unit,
    target_timestamp_unit,
    onset_col=app.config["ONSET_COL"],
    offset_col=app.config["OFFSET_COL"],
    code_col=app.config["CODE_COL"]):
    '''read csv/excel files'''
    error_message = ""
    if fn.endswith("csv"):
        df = pd.read_csv(filepath)
        cols = list(df.columns)
        cols[:4] = ["index", onset_col, offset_col, code_col]
        df.columns = cols
        df = df[["index", onset_col, offset_col, code_col]]
    else:
        try:
            # xlsx file coded using frame as time unit doesn't have header
            df = pd.read_excel(filepath, header=None)
            df.columns = [code_col, onset_col, offset_col]
            df = df.reset_index()
        except:
            df = None
            error_message = "%s cannot be openned,"\
                            " please make sure it is csv/xlsx file format\n"\
                            "%s"%(fn, traceback.format_exc())
    if df is not None :
        df = df.fillna(0)
        if (original_timestamp_unit == "milisecond") \
            and (target_timestamp_unit == "frame"):
            df = convert_milisecond_to_frame(df)
        if (original_timestamp_unit == "frame") \
            and (target_timestamp_unit == "milisecond"):
            df = convert_frame_to_milisecond(df)
        
    return df, error_message


def check_eligible_codes_match(codes, eligible_codes):
    eligible_codes = [c.strip() for c in eligible_codes.split(",")]
    return set(codes).issubset(set(eligible_codes))


def check_trial_begin_and_end_are_paired(df, code_col=app.config["CODE_COL"],
    begining_code=app.config["BEGIN_CODE"], ending_code=app.config["END_CODE"]):
    dft = df[df[code_col].isin([begining_code, ending_code])]
    codes = list(dft[code_col].values)
    code_stack = []
    try:
        for code in codes:
            if code == begining_code:
                # everytime a new trial begins,
                # we must found previous trial has a paired begining and end
                assert len(code_stack) == 0, "Trial has unpaired begining and end"
                code_stack.append(code)
            if code == ending_code:
                code_stack.pop()
                # everytime a trail ends
                # it must pop up its begining, leave the code stack empty
                assert len(code_stack) == 0, "Trial has unpaired begining and end"
        # at the end, there should not be any unpaired beginings
        assert len(code_stack) == 0, "Trial has unpaired begining and end"
    except:
        return False, codes.count(begining_code)
    return True, codes.count(begining_code)


def check_non_begining_nor_end_code_has_on_off_set(df, 
    code_col=app.config["CODE_COL"],
    begining_code=app.config["BEGIN_CODE"], ending_code=app.config["END_CODE"]):

    dft = df[(df[code_col] != begining_code) & (df[code_col] != ending_code)]
    no_onset_value_rows = dft[dft["onset"] == 0]
    no_offset_value_rows = dft[dft["offset"] == 0]
    if (len(no_onset_value_rows) == 0) and (len(no_offset_value_rows) == 0):
        return True
    else:
        return False


def update_status_df(file_id, column, value, session, 
                     records_key="eyegazecleaner_records"):
    status_records = session.get(records_key, [])
    status_df = pd.DataFrame.from_dict(status_records)
    status_record_ind = status_df[status_df["ID"] == file_id].index
    status_df.loc[status_record_ind, column] = value
    status_records = status_df.to_dict("records")
    session[records_key] = status_records


def run_quality_check(dft, file_id, session,
                      eligible_codes, expected_num_trials,
                      begin_code, end_code,
                      code_col=app.config["CODE_COL"]):
    codes = list(dft[code_col].value_counts().index)
    eligible_codes_okay = check_eligible_codes_match(codes, 
        eligible_codes)
    session["%s_eligible_codes_okay"%file_id] = eligible_codes_okay

    paired_begin_and_end, num_trials\
        = check_trial_begin_and_end_are_paired(dft, 
                code_col=code_col,
                begining_code=begin_code, 
                ending_code=end_code)
    num_trials_match_expectation = (num_trials == expected_num_trials)
    session['%s_paired_begin_and_end'%file_id] = paired_begin_and_end
    session['%s_num_trials'%file_id] = num_trials
    session['%s_num_trials_match_expectation'%file_id] \
       = num_trials_match_expectation

    onset_offset_check = check_non_begining_nor_end_code_has_on_off_set(dft, 
                code_col=code_col,
                begining_code=begin_code, 
                ending_code=end_code)
    session['%s_onset_offset_check'%file_id] = onset_offset_check

    overall_quality =  eligible_codes_okay \
                    & paired_begin_and_end \
                    & onset_offset_check \
                    & num_trials_match_expectation
    session["%s_quality"%file_id] = overall_quality
    return overall_quality


def rearrange_codes(codes, default_begin_code=app.config["BEGIN_CODE"], 
    default_end_code=app.config["END_CODE"]):
    codes = sorted(codes)
    if default_end_code in codes:
        codes.remove(default_end_code)
        codes.insert(0, default_end_code)
    if default_begin_code in codes:
        codes.remove(default_begin_code)
        codes.insert(0, default_begin_code)
    return codes


def get_trial_summary(df, code_col=app.config["CODE_COL"],
                      begin_code=app.config["BEGIN_CODE"],
                      end_code=app.config["END_CODE"],
                      onset_col=app.config["ONSET_COL"],
                      offset_col=app.config["OFFSET_COL"],
                      code_meaning_dict = app.config["CODE_MEANING_DICT"]):
    
    trial_id_col = app.config["TRIAL_ID_COL"]
    df = assign_trial_id(df, code_col, begin_code)
    grped = df.groupby(trial_id_col)
    eligible_codes = list(df[code_col].value_counts().index)
    non_begin_end_codes = [c for c in eligible_codes \
                           if c not in [begin_code, end_code]]
    records = []
    ordered_cols = [trial_id_col, begin_code, end_code]
    for t in ["longest", "total"]:
        for c in non_begin_end_codes:
            ordered_cols\
                .append("%s.%s"%(code_meaning_dict.get(c, c), t))
    ordered_cols.extend(["total.screen.look", 
                         "total.trial.length", 
                         "attention.entire.trial"])

    for trial_id, grp in grped:
        begin_time = grp[grp[code_col] == begin_code][onset_col].values[0]
        end_time = grp[grp[code_col] == end_code][onset_col].values[0]
        record = {
                    trial_id_col: trial_id, 
                    begin_code: begin_time,
                    end_code: end_time,
                    "total.trial.length": end_time - begin_time
                 }
        total_screen_look = 0
        for c in non_begin_end_codes:
            t = grp[grp[code_col] == c]
            t["duration"] = t[offset_col] - t[onset_col]
            record["%s.longest"%code_meaning_dict.get(c, c)] = t["duration"].max()
            record["%s.total"%code_meaning_dict.get(c, c)] = t["duration"].sum()
            total_screen_look += t["duration"].sum()
        record["total.screen.look"] = total_screen_look
        record["attention.entire.trial"] \
            = total_screen_look / record["total.trial.length"] * 100
        records.append(record)
    summary_df = pd.DataFrame.from_dict(records)
    summary_df = summary_df.fillna(0)
    summary_df = summary_df[ordered_cols]
    return summary_df


def to_readable_ts(row, begin_code_col, unit="milisecond"):
    t = row[begin_code_col]
    if unit == "frame":
        # covert to milisecond
        t = t * (100/3)
    t = t / 1000
    return str(timedelta(seconds=t))


def run_trial_summary_comparison_two(records1, unit1, 
                                     records2, unit2, begin_code):
    df1 = pd.DataFrame.from_dict(records1)
    df2 = pd.DataFrame.from_dict(records2)

    if unit1 != unit2:  # convert coder2 to use the same unit as coder1
        if (unit1 == "frame") \
                and (unit2 == "milisecond"):
                df2 = convert_milisecond_to_frame(df2, filetype="trial_summary")
        else:
            df2 = convert_frame_to_milisecond(df2, filetype="trial_summary") 

    dft = df1.join(df2, lsuffix='.1', rsuffix='.2')
    ordered_cols = []
    for c in df1.columns:
        if c == app.config["TRIAL_ID_COL"]:
            dft[app.config["TRIAL_ID_COL"]] = dft["%s.1"%c]
            ordered_cols.append(c)
        else:
            if c == "attention.entire.trial":
                ordered_cols.extend(["%s.1"%c, "%s.2"%c])
            else:
                dft["%s.diff"%c] = np.abs(dft["%s.1"%c] - dft["%s.2"%c])
                ordered_cols.extend(["%s.1"%c, "%s.2"%c, "%s.diff"%c])

    if unit1 == "milisecond":
        dft["%s.hh:mm:ss.timestamp"%begin_code] \
            = dft.apply(to_readable_ts, 
                        args=("%s.1"%begin_code, "milisecond"), axis=1)
    else:
        dft["%s.hh:mm:ss.timestamp"%begin_code] \
            = dft.apply(to_readable_ts, 
                        args=("%s.1"%begin_code, "frame"), axis=1)
    ordered_cols.insert(1, "%s.hh:mm:ss.timestamp"%begin_code)
            
    dft = dft[ordered_cols]
    diff_col_indices = [i for i, c in enumerate(dft.columns) \
                        if c.endswith(".diff")]
    
    return dft.to_dict("records"), dft.columns, diff_col_indices


def has_discrepancy(row, diff_cols, threshold):
    for c in diff_cols:
        if row[c] > threshold:
            return 1
    return 0


def highlight_compare_two_discrepancy_cell(x, threshold, 
        color=app.config["FAILED_TWOWAYCOMPARE_COLOR"]):
    return np.where(x > threshold, f"background-color: {color};", None)


def highlight_compare_two_discrepancy_trialid(x, l, 
        color=app.config["FAILED_TWOWAYCOMPARE_COLOR"]):
    return np.where(x.isin(l), f"background-color: {color};", None)


def highlight_compare_two_discrepancy(df, threshold, 
                                      trial_id_col=app.config["TRIAL_ID_COL"]):
    diff_cols = [c for c in df.columns if c.endswith(".diff")]
    df["has_discrepancy"] = df.apply(has_discrepancy, 
                                    args=(diff_cols, threshold),
                                    axis=1)
    # trail id is 1 indexed
    has_discrepancy_trial_ids = np.where(df["has_discrepancy"] == 1)[0] + 1
    df = df.style.apply(highlight_compare_two_discrepancy_cell, 
                        threshold=threshold, subset=diff_cols)\
                  .apply(highlight_compare_two_discrepancy_trialid, 
                        l=has_discrepancy_trial_ids, subset=trial_id_col)
    return df


def get_overlap_frac(row, begin, end, begin_code="B", end_code="S"):
    latest_begin = max(row[begin_code], begin)
    earliest_end = min(row[end_code], end)
    overlap = earliest_end - latest_begin
    if overlap <= 0:
        return 0
    else:
        return overlap / (end - begin)
    

def add_coder3_to_paircomparison(df12, df3,
    begin_code="B",
    end_code="S",
    trial_id_col="trial_id"):
    
    df3 = df3.copy()
    
    coder3_trial_id_in_coder1 = []
    to_fix_trials = []
    for ind, row in df3.iterrows():
        begin = row["B"]
        end = row["S"]
        v1 = df12.apply(get_overlap_frac, args=(begin, end, begin_code+'.1', end_code+'.1'), axis=1)
        which_coder1_trial = df12.iloc[np.argmax(v1)][trial_id_col]
        which_coder1_trial_ind = np.argmax(v1)
        coder3_trial_id_in_coder1.append(int(which_coder1_trial))
        to_fix_trials.append((int(which_coder1_trial_ind), int(which_coder1_trial)))

        v2 = df12.apply(get_overlap_frac, args=(begin, end, begin_code+'.2', end_code+'.2'), axis=1)
        which_coder2_trial = df12.iloc[np.argmax(v2)][trial_id_col]

        if (which_coder1_trial != which_coder2_trial):
            error_message = "CANNOT PLACE THE CODER 3 TRIAL "\
                  "IN CODER 1 and 2 coding files,"\
                  " make sure coder 1-3 are coding the same trials. Details:"\
                  "coder3 trial %s map to trial %s in coder1; map to trial %s in coder2"\
                    %(ind, which_coder1_trial, which_coder2_trial)
            return False, error_message, None, [], {} 
        if np.max(v1) == 0:
            error_message = "CANNOT PLACE THE CODER 3 TRIAL "\
                  "IN CODER 1 and 2 coding files,"\
                  " make sure coder 1-3 are coding the same trials. Details:"\
                  "coder3 trial %s has no overlap to trials in coder1"%ind
            return False, error_message, None, [], {} 

    coder3_trial_id_lookup = dict(zip(df3[trial_id_col], 
                                      coder3_trial_id_in_coder1))
    df3[trial_id_col] = coder3_trial_id_in_coder1
    df3.columns = [c+".3" if c != trial_id_col else c for c in df3.columns]
    dft = pd.merge(df12, df3, how="left", on=trial_id_col)
    
    merged_ordered_cols = ["%s%s"%(re.findall("(.*)[0-9]", c)[0], coder) \
                           if (c != trial_id_col) and (not c.endswith("diff") \
                               and (not c.endswith(".timestamp"))) else c \
                           for c in df12.columns for coder in [1,2,3]]
    res  = []
    _ = [res.append(x) for x in merged_ordered_cols if (x not in res) and (x in dft.columns)]
    merged_ordered_cols = res
    dft = dft[merged_ordered_cols]
    return True, "", dft, to_fix_trials, coder3_trial_id_lookup


def highlight_coder3_resolution(x, l, color="#EAE7B1"):
    color_list = np.array([None]*len(x))
    color_list[l] = f"background-color: {color};"
    return color_list


def get_coder_with_most_agreement(row, agreement_frac_thresh=0.5):
    counts = [0, 0, 0]

    ttl = 0
    row = row.to_dict()
    for col, winners in row.items():
        if ("similarity_winner" in col) and (winners != "NA"):
            ttl += 1
            for coder in winners:
                counts[coder-1] += 1  # coder 1 indexing

    counts = np.array(counts)
    agreement_fracs = counts / ttl 
    winner = np.argwhere(agreement_fracs == np.amax(agreement_fracs))\
                .squeeze(axis=-1) + 1 # coder 1 indexing
    winner = list(winner)
    winner_count = np.max(counts)
    highest_agreement_frac = np.max(agreement_fracs)

    if ttl == 0:
        # no discrepancy, in this case, highest agreement frac is nan
        # but trial should be considered usable
        trial_is_usable = True
    else:
        if highest_agreement_frac >= agreement_frac_thresh:
            trial_is_usable = True
        else:
            trial_is_usable = False
    return trial_is_usable, winner, ttl, winner_count, highest_agreement_frac


def threeway_resolution(dft, df3, to_fix_trials, threshold, 
                        trial_id_col=app.config["TRIAL_ID_COL"]):
    '''
    :param to_fix_trials: list of tuples, [(index, trial_id)], ex. to_fix_trials = [(13, 14)] 
    '''
    to_compare_cols = [c for c in df3.columns if c not in [trial_id_col,'attention.entire.trial']]
    resolution_records = []
    for ind, trial_id in to_fix_trials:
        row = dft[dft[trial_id_col] == trial_id].iloc[0]
        record = {"index":ind, trial_id_col : trial_id}

        for c in to_compare_cols:
            c1 = row[c+".1"]
            c2 = row[c+".2"]
            c3 = row[c+".3"]
            diff_12 = np.abs(c2 - c1)
            if diff_12 > threshold: 
                diff_1 = np.abs(c3 - c1)
                diff_2 = np.abs(c3 - c2)

                if diff_1 < diff_2:
                    if diff_1 <= threshold:
                        record["similarity_winner.%s"%c] = [1,3]
                    else:
                        record["similarity_winner.%s"%c] = []

                elif diff_1 == diff_2:
                    if diff_1 == diff_2 == 0:
                        record["similarity_winner.%s"%c] = [1,2,3]
                    elif diff_1 <= threshold:
                        record["similarity_winner.%s"%c] = [1,2]
                    else:
                        record["similarity_winner.%s"%c] = []

                else:
                    if diff_2 <= threshold:
                        record["similarity_winner.%s"%c] = [2,3]
                    else:
                        record["similarity_winner.%s"%c] = []

            else:
                record["similarity_winner.%s"%c] = "NA"

        resolution_records.append(record)

    resolution_df = pd.DataFrame(resolution_records)
    resolution_df[["trial_is_usable", "coder_with_most_agreement", "num_discrepancy", 
               "num_agreement_by_the_winner_coder", "agreement_percentage"]] \
        = resolution_df.apply(get_coder_with_most_agreement,
                              axis=1, result_type="expand")
    return resolution_df


def colorcode_threeway_comparison(dft, resolution_df,
    threshold,
    similarity_winner_coder_color=app.config["SIMILARITY_WINNER_CODER_COLOR"],
    failed_threewaycompare_color=app.config["FAILED_THREEWAYCOMPARE_COLOR"]):

    dft["which_coder"] = 1
    dft["trial_is_usable"] = True
    for _, row in resolution_df.iterrows():
        for c in row.index:
            if c == "index":
                coder_with_most_agreement = row["coder_with_most_agreement"]
                if len(coder_with_most_agreement):
                    coder_with_most_agreement = min(coder_with_most_agreement)
                else:
                    coder_with_most_agreement = None
                dft.at[row[c], "which_coder"] = coder_with_most_agreement
                dft.at[row[c], "trial_is_usable"] =\
                    row["trial_is_usable"]

    t = highlight_compare_two_discrepancy(dft, threshold)
    for _, row in resolution_df.iterrows():
        style_col_subsets = []
        for c in row.index:
            if c.startswith("similarity_winner") and (row[c] != "NA"):
                base_col_name = c.split("similarity_winner.")[1]
                coders = row[c]
                if len(coders):
                    style_col_subsets = ["%s.%s"%(base_col_name, coder) for coder in coders]
                    # light green
                    t = t.apply(highlight_coder3_resolution, l=[row["index"]], 
                                color=similarity_winner_coder_color,
                                subset=style_col_subsets)
                else: # no coders match 3rd coder
                    style_col_subsets = ["%s.%s"%(base_col_name, coder) for coder in [1,2,3]]
                    # red
                    t = t.apply(highlight_coder3_resolution, l=[row["index"]], 
                                color=failed_threewaycompare_color,
                                subset=style_col_subsets)

    return t


def threeway_comparison(records12, unit1, records3, unit3, threshold,
                        trial_id_col=app.config["TRIAL_ID_COL"]):
    df12 = pd.DataFrame.from_dict(records12)
    df3 = pd.DataFrame.from_dict(records3)

    if unit1 != unit3:  # convert coder2 to use the same unit as coder1
        if (unit1 == "frame") \
                and (unit3 == "milisecond"):
                df3 = convert_milisecond_to_frame(df3, filetype="trial_summary")
        else:
            df3 = convert_frame_to_milisecond(df3, filetype="trial_summary") 

    add_trial3_status, error_message, dft, to_fix_trials, coder3_trial_id_lookup \
        = add_coder3_to_paircomparison(df12, df3)
    if add_trial3_status:
        resolution_df = threeway_resolution(dft, df3, to_fix_trials, threshold)
        dft = colorcode_threeway_comparison(dft, resolution_df,
                                            threshold)
        return True, error_message, (dft, resolution_df, coder3_trial_id_lookup)
    else:
        return False, error_message, ()


def combine_coding(compare_records, 
                   coder1_records, coder1_begin_code, code1_filename,
                   coder2_records, coder2_begin_code, code2_filename,
                   coder3_records=[], 
                   coder3_begin_code=app.config["BEGIN_CODE"],
                   code3_filename="NA",
                   coder3_trial_id_lookup={},
                   trial_id_col=app.config["TRIAL_ID_COL"]):
    compare_df = pd.DataFrame.from_dict(compare_records)
    if "which_coder" not in compare_df.columns:   # twoway coding
        compare_df["which_coder"] = 1
    if "trial_is_usable" not in compare_df.columns:
        compare_df["trial_is_usable"] = True
    df1 = pd.DataFrame.from_dict(coder1_records)
    df2 = pd.DataFrame.from_dict(coder2_records)
    df3 = pd.DataFrame.from_dict(coder3_records)
    df1 = assign_trial_id(df1, begin_code=coder1_begin_code)
    df2 = assign_trial_id(df2, begin_code=coder2_begin_code)
    if not df3.empty:
        df3 = assign_trial_id(df3, begin_code=coder3_begin_code)
        df3[trial_id_col] = [coder3_trial_id_lookup.get(i, i) \
                            for i in df3[trial_id_col].values]
    dfs = {1:df1, 2:df2, 3:df3}
    filenames= {1: code1_filename, 2:code2_filename, 3:code3_filename}
    dfts = []
    for _, row in compare_df.iterrows():
        if row["trial_is_usable"]:
            t = dfs[row["which_coder"]]
            t["from_coder"] = row["which_coder"]
            t["from_filename"] = filenames[row["which_coder"]]
            t = t[t["trial_id"] == row["trial_id"]]
            dfts.append(t)
    dft = pd.concat(dfts)
    cols = list(dft.columns)
    cols.remove(trial_id_col)
    cols.insert(1, trial_id_col)
    dft = dft[cols]

    return dft


def read_custom_combine_data(fn, filepath):
    '''read csv/excel files'''
    error_message = ""
    if fn.endswith("csv"):
        df = pd.read_csv(filepath)
    else:
        try:
            # xlsx file coded using frame as time unit doesn't have header
            df = pd.read_excel(filepath)
        except:
            df = None
            error_message = "%s cannot be openned,"\
                            " please make sure it is csv/xlsx file format\n"\
                            "%s"%(fn, traceback.format_exc())
    return df, error_message