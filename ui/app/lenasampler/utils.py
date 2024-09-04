import os
import re
import uuid
import shutil
import pandas as pd
import numpy as np
import zipfile
from zipfile import ZipFile
from tqdm import tqdm
from moviepy.editor import *
from app.lenasampler.detect_its_gap import detect_gap, parse_time_row


def get_random_id():
    return str(uuid.uuid4())[:8]


def get_id_prefix(fn):
    id_prefix = re.findall("[A-Z]+[0-9]+_", fn)
    if len(id_prefix):
        return "%s_"%id_prefix[0]
    else:
        return ""
        

def preprocess(fn, itsfilecol, starttimecol, endtimecol, durationcol):
    df = pd.read_csv(fn).reset_index()
    df['StartTime_ts'] = df.apply(parse_time_row, args=(starttimecol, ), axis=1)
    df['EndTime_ts'] = df.apply(parse_time_row, args=(endtimecol, ), axis=1)
    
    for its_file in df[itsfilecol].unique():
        dft = df[df[itsfilecol]==its_file].copy()
        l = np.array(dft.sort_values('StartTime_ts').index)
        ind = np.where(sorted(l) != l)[0]
        assert len(ind) == 0, \
            f"{its_file} time is not sorted. Please make sure the file is correct!"
        df.loc[dft.index, 'RelativeStart'] = dft[durationcol].cumsum() - dft[durationcol]
    df['RelativeStart'] = df['RelativeStart'].astype(int)
    return df


def remove_audio_fn_prefix(fn):
    """ 
    There will be two types of wav filenames
    Type 1: 20210311_135447_010263_2.wav. This is the default filename 
    LENA generated. No action needed.
    Type 2: M001_20210311_135447_010263_2.wav. In this type, a id prefix is 
    mannually added. The id prefix follows [A-Z]+[0-9]+_ pattern. In this 
    type, the id prefix needs to be removed, ie, 20210311_135447_010263_2.wav
    """
    id_prefix = re.findall("[A-Z]+[0-9]+_", fn)
    if len(id_prefix):
        return "_".join(fn.split("_")[1:])
    else:
        return fn


def get_wav_its_dict(audio_dir):
    '''
    wav_its_dict map its file to corresponding wav file
    '''
    audio_files = [ fn for fn in os.listdir(audio_dir) \
                    if (fn.endswith("wav") and not fn.startswith("._"))]
    return {remove_audio_fn_prefix(fn.replace(".wav", ".its")):fn \
                        for fn in audio_files}


def its_wav_match_quality_check(its_file_names, audio_dir):
    
    # convert wav to its file; make {wav_corresponding_its_fn: wav_fn} dict
    wav_its_dict = get_wav_its_dict(audio_dir)
    missing_files = sorted(list(set(its_file_names) - wav_its_dict.keys()))
    extra_files = sorted(list(wav_its_dict.keys() - set(its_file_names)))
    matched_files = sorted(list(wav_its_dict.keys()&set(its_file_names)))
    # its_to_wav_files = ["%s%s"%(get_id_prefix(fn), 
    #                             fn.replace(".its", ".wav")) \
    #                     for fn in its_file_names]
    # missing_files = sorted(list(set(its_to_wav_files) - set(audio_files)))
    # missing_files = [remove_audio_fn_prefix(fn.replace(".wav", ".its"))\
    #                  for fn in missing_files]
    # extra_files = sorted(list(set(audio_files) - set(its_to_wav_files)))
    # matched_files = sorted(list(set(audio_files) & set(its_to_wav_files))) 
    extra_files = [wav_its_dict[fn] for fn in extra_files]
    matched_files = [wav_its_dict[fn] for fn in matched_files]

    if (len(missing_files) == 0) and (len(extra_files) == 0):
        is_perfect_match = True
    else:
        is_perfect_match = False
    return missing_files, extra_files, matched_files, is_perfect_match


def get_audio_duration(fn):
    ''' Measure the duration of a wav file
    '''
    a = AudioFileClip(fn)
    return int(a.duration)


def check_audio_duration_match(df, audio_dir, matched_files, 
                       missing_files, extra_files,
                       itsfile_col, duration_col):
    '''
        :params df: dataframe, should contain the its file name 
                    column and duration column
    '''
    records = []
    for fn in matched_files:
        itsfilename = fn.replace(".wav", ".its")
        itsfilename = remove_audio_fn_prefix(itsfilename)
        dft = df[df[itsfile_col] == itsfilename]
        its_duration = dft[duration_col].sum()
        audio_filepath = os.path.join(audio_dir, fn)
        audio_duration = get_audio_duration(audio_filepath)
        diff = audio_duration - its_duration
        if diff == 0:
            note = "Perfect match"
        else:
            note = "The duration of its file and wav is not the same."
        t_record = {"Filename": itsfilename,
                    "Type": "Matched",
                    "its duration (s)": its_duration,
                    "wav duration (s)": audio_duration,
                    "note": note}
        records.append(t_record)
    
    for fn in missing_files:
        dft = df[df[itsfile_col] == fn]
        its_duration = dft[duration_col].sum()
        t_record = {"Filename": fn, 
                    "Type": "No matching wav file",
                    "its duration (s)": its_duration,
                    "wav duration (s)": "NA",
                    "note": "No matching wav file found for this its file"}
        records.append(t_record)
    
    for fn in extra_files:
        audio_filepath = os.path.join(audio_dir, fn)
        audio_duration = get_audio_duration(audio_filepath)
        t_record = {"Filename": fn, 
                    "Type": "No matching its file",
                    "its duration (s)": "NA",
                    "wav duration (s)": audio_duration,
                    "note": "No matching its file found for this wav file"}
        records.append(t_record)

    return records


def run_quality_check(records, audio_dir, itsfile_col, duration_col):
    df = pd.DataFrame(records)
    its_gap_records, n_itsfile = detect_gap(df,  threshold=1, its_col=itsfile_col)
    if len(its_gap_records):
        its_with_gaps = ",".join(list(its_gap_records["ItsFile"].unique()))
    else:
        its_with_gaps = ""
        
    its_file_names = df[itsfile_col].values
    # does the its file column correspond to the audio wav file?
    missing_files, extra_files, matched_files, is_perfect_match \
        = its_wav_match_quality_check(its_file_names, audio_dir)
    quality_records = [
        {"Item": "ITS files perfect match WAV files",
         "Value": is_perfect_match, "Notes": ''},
        {"Item": "ITS files without corresponding WAV files",
         "Value": ", ".join(missing_files), "Notes": ''},
        {"Item": "WAV files without corresponding ITS files",
         "Value": ", ".join(extra_files), "Notes": ''},
        {"Item": "Matched ITS and WAV files",
         "Value": ", ".join(matched_files), "Notes": ''},
        {"Item": "ITS file with time gaps",
         "Value": its_with_gaps, 
         "Notes": 'Need to be extra careful at checking the video quality for segments from this file'
        }
    ]
    dft_summary = pd.DataFrame(quality_records)
    dft_summary = dft_summary.reset_index()

    # Does the length of a wav file correspond to 
    # the sum of the length of the segments?
    duration_match_records = check_audio_duration_match(df, audio_dir, 
        matched_files, missing_files, extra_files, itsfile_col, duration_col)
    dft_per_file = pd.DataFrame(duration_match_records)
    dft_per_file = dft_per_file.reset_index()
    matched_itsfiles = [remove_audio_fn_prefix(fn.replace(".wav", ".its"))\
                     for fn in matched_files]

    return dft_summary, dft_per_file, matched_itsfiles, is_perfect_match


def extract_from_audio_file(its_file, audiodir, start, duration,
                            abs_ts, outdir, index):
    wav_its_dict = get_wav_its_dict(audiodir)
    audio_file =  wav_its_dict.get(its_file, "")
    audio_filepath = os.path.join(audiodir, audio_file)
    outfn = audio_file.replace(".wav", "_AbsStart_%s_RelStart_%s_Duration_%s.wav"\
                                %(abs_ts, start, duration))
    outfn = os.path.join(outdir, "%s_%s"%(index, outfn))
    a = AudioFileClip(audio_filepath)
    segment = a.subclip(start, start+duration)
    segment.write_audiofile(outfn)
    return outfn


def prepare_audio_files(df, df_ori, audiodir, outdir, idprefix,
                        itsfilecol, starttimecol, endtimecol, durationcol):
    idprefix = idprefix.strip("_")  # remove any trailing _
    df = df.sort_values("index")
    if os.path.exists(outdir):  # remove existing outdir if there is one
        shutil.rmtree(outdir, ignore_errors=True)
    else:
        os.makedirs(outdir)

    relative_start_time = []
    outfns = []
    for _, row in df.iterrows():
        its_file = row[itsfilecol]
        segment_relative_start = row['RelativeStart']
        segment_start = row['StartTime_ts']
        duration = row[durationcol]
        relative_start_time.append(segment_relative_start) 
        outfn = extract_from_audio_file(its_file, audiodir, 
                                        segment_relative_start, duration,
                                        str(segment_start), outdir, row["index"]) 
        outfns.append(os.path.basename(outfn))

    df["segment_relative_start_time"] = relative_start_time
    df["segment_filename"] = outfns
    df.to_csv(os.path.join(outdir, "%s_SampledAudioSegmentsMetadata.csv"%idprefix))
    return df


def zipdir(path, ziph):
    print("\nZipping files ...")
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        if len(files):
            for file in tqdm(files):
                ziph.write(os.path.join(root, file))
