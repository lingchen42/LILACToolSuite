import traceback
import numpy as np
import pandas as pd
from app import app


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


def assign_trial_id(df, code_col=app.config["CODE_COL"], 
                    begin_code=app.config["BEGIN_CODE"]):
    codes = df[code_col].values
    trial_ids = []
    trial_id = 0
    for c in codes:
        if c == begin_code:
            trial_id += 1
        trial_ids.append(trial_id)
    df[app.config["TRIAL_ID_COL"]] = trial_ids
    return df


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


def run_trial_summary_comparison_two(records1, unit1, records2, unit2):
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
            
    dft = dft[ordered_cols]
    diff_col_indices = [i for i, c in enumerate(dft.columns) \
                        if c.endswith(".diff")]
    
    return dft.to_dict("records"), dft.columns, diff_col_indices


def has_discrepancy(row, diff_cols, threshold):
    for c in diff_cols:
        if row[c] > threshold:
            return 1
    return 0


def highlight_compare_two_discrepancy_cell(x, threshold, color="red"):
    return np.where(x > threshold, f"background-color: {color};", None)


def highlight_compare_two_discrepancy_trialid(x, l, color="red"):
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


def find_trial_with_max_overlap(coder1_summary, coder2_summary, coder3_summary):
    pass