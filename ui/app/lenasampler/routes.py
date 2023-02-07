import os
import uuid
import shutil
import tempfile
import traceback
import pandas as pd
import zipfile
from zipfile import ZipFile
from werkzeug.utils import secure_filename
from flask import redirect, render_template, url_for, request, session, Response
from app import app
from app.lenasampler import bp
from app.lenasampler.forms import *
from app.lenasampler.utils import *


@bp.route('/data', methods=['GET', 'POST'])
def data():
    form = DataInput()
    records = session.get('records', [])
    columns = session.get('columns', [])
    filename = session.get('filename', '')
    audio_dir = session.get('audio_dir', '')
    quality_check_status = session.get('quality_check_status', 'Unknown')
    status_records = []
    status_columns = []

    if form.validate_on_submit():
        session["audio_dir"] = form.audio_dir.data
        if not os.path.exists(form.audio_dir.data):
            return render_template("error.html",
                message="Audio Directory %s does not exist. Please double check."\
                        %form.audio_dir.data)

        fn = form.fn.data
        if fn:
            filename = secure_filename(fn.filename)
            session["filename"] = filename 
            idprefix = get_id_prefix(filename)
            if not idprefix:
                idprefix = get_random_id()
            session["idprefix"] = idprefix
            with tempfile.NamedTemporaryFile() as tmp:
                fn.save(tmp.name)
                try:
                    dft = pd.read_csv(tmp.name) 
                    dft = dft.reset_index()
                    columns = list(dft.columns)
                    records = dft.to_dict("records")
                    session['columns'] = columns
                    session['records'] = records
                except Exception as e:
                    return render_template("error.html", 
                                            message=traceback.format_exc())

        return redirect(url_for('lenasampler.view_data'))

    status_df = pd.DataFrame({
        "Current Items": ["Filename", "Audio Directory", 
                 "Pass Quality Check"],
        "Value": [filename, audio_dir, quality_check_status],
        "Notes": ['', '', '']
    })
    status_df = status_df.reset_index()
    status_records = status_df.to_dict("records")
    status_columns = status_df.columns
    
    return render_template("lenasampler/data.html",
                            form=form,
                            columns=status_columns,
                            records=status_records)


@bp.route('/view_data', methods=['GET', 'POST'])
def view_data():
    records = session.get('records', [])
    columns = session.get('columns', [])
    return render_template("lenasampler/view_data.html",
                            columns=columns,
                            records=records)


@bp.route('/quality_check', methods=['GET', 'POST'])
def quality_check():
    records = session.get('records', [])
    quality_summary_columns = session.get("quality_summary_columns", [])
    quality_summary_records = session.get("quality_summary_records", [])
    quality_perfile_columns = session.get("quality_perfile_columns", [])
    quality_perfile_records = session.get("quality_perfile_records", [])
    audio_dir = session.get('audio_dir', None)
    itsfilecol = app.config["ITS_FILENAME_COL"]
    durationcol = app.config["DURATION_COL"]
    dft_summary, dft_per_file, matched_itsfiles, is_perfect_match \
        = run_quality_check(records, audio_dir, itsfilecol, durationcol)
    quality_summary_columns = dft_summary.columns
    quality_summary_records = dft_summary.to_dict("records")
    quality_perfile_columns = dft_per_file.columns
    quality_perfile_records = dft_per_file.to_dict("records")
    session["matched_itsfiles"] = matched_itsfiles
    session["quality_summary_columns"] = quality_summary_columns
    session["quality_summary_records"] = quality_summary_records
    session["quality_perfile_columns"] = quality_perfile_columns
    session["quality_perfile_records"] = quality_perfile_records
    session['quality_check_status'] = is_perfect_match

    return render_template("lenasampler/quality_check.html",
                            columns1=quality_summary_columns,
                            records1=quality_summary_records,
                            columns2=quality_perfile_columns,
                            records2=quality_perfile_records)


@bp.route('/filter', methods=['GET', 'POST'])
def filter():
    columns = session.get('columns', [])
    records = session.get('records', [])
    matched_itsfiles = session.get('matched_itsfiles', [])
    dft = pd.DataFrame.from_records(records)
    records_after_filtering = session.get('records_after_filtering', records)
    selected_itsfiles = session.get('selected_itsfiles', matched_itsfiles)
    form = FilterForm(matched_itsfiles)
    if request.method == "GET":
        for col in app.config["DEFAULT_FILTER_NUM_COLUMNS"]:
            minv = session.get("%s_min_value"%col, dft[col].min())
            maxv = session.get("%s_max_value"%col, dft[col].max())
            setattr(getattr(form, "%s_min_value"%col), "data", minv)
            setattr(getattr(form, "%s_max_value"%col), "data", maxv)
            setattr(getattr(form, "itsfiles"), "data", selected_itsfiles)

    if form.validate_on_submit():
        # filter its files
        selected_itsfiles = form.itsfiles.data
        session["selected_itsfiles"] = selected_itsfiles
        dft = dft[dft[app.config["ITS_FILENAME_COL"]].isin(selected_itsfiles)]

        # filter segments
        for col in app.config["DEFAULT_FILTER_NUM_COLUMNS"]:
            minv = getattr(getattr(form, "%s_min_value"%col), "data")
            session["%s_min_value"%col] = minv
            dft = dft[dft[col] >= minv]
            maxv = getattr(getattr(form, "%s_max_value"%col), "data")
            session["%s_max_value"%col] = maxv
            dft = dft[dft[col] <= maxv]
        records_after_filtering = dft.to_dict("records")
        session["records_after_filtering"] = records_after_filtering

    return render_template("lenasampler/filter.html",
                           form=form,
                           columns=columns,
                           records=records_after_filtering)


@bp.route("/sample1", methods=['GET', 'POST'])
def sample1():
    records = session.get('records', [])
    dft = pd.DataFrame.from_records(records)
    num_cols = list(dft._get_numeric_data().columns)
    sampling_cols_form = SamplingColsForm(num_cols)
    if request.method == "GET":
        selected_sampling_cols = session.get("sampling_criteria_cols", 
                                        app.config["SAMPLING_CRITERIA_COLS"])
        sampling_cols_form.samplingcols.data \
            = selected_sampling_cols  

    if sampling_cols_form.validate_on_submit():
        session["sampling_criteria_cols"] = sampling_cols_form.samplingcols.data
        return redirect(url_for("lenasampler.sample2"))

    return render_template("lenasampler/sampling.html", 
                            form=sampling_cols_form,
                            columns=[],
                            records=[])


@bp.route("/sample2", methods=['GET', 'POST'])
def sample2():
    records = session.get('records_after_filtering', [])
    sampled_records = session.get('sampled_records', [])
    columns = session.get('columns', [])
    dft = pd.DataFrame.from_records(records)
    sampling_criteria_cols = session.get("sampling_criteria_cols", [])

    class SamplingForm(FlaskForm):
        target_num_segments = IntegerField("Target number of segments", 
            default=12, validators=[DataRequired(), NumberRange(min=0, max=len(dft))])
        random_seed = IntegerField("Sampling Seed (for reproducibility)", 
            default=1, validators=[DataRequired(), NumberRange(min=0)])

    for col in sampling_criteria_cols :
        setattr(SamplingForm, "%s_min_value"%col, 
            IntegerField("%s min value (>=, %s median %.1f, mean %.1f)"\
                        %(col, col, dft[col].median(), dft[col].mean())))
        setattr(SamplingForm, "%s_max_value"%col,
            IntegerField("%s max value (<=, %s median %.1f, mean %.1f)"\
                        %(col, col, dft[col].median(), dft[col].mean())))
    setattr(SamplingForm, "submit", SubmitField('Confirm'))
    form = SamplingForm()

    if request.method == "GET":
        for col in sampling_criteria_cols:
            minv = session.get("sampling_%s_min_value"%col, dft[col].min())
            maxv = session.get("sampling_%s_max_value"%col, dft[col].max())
            setattr(getattr(form, "%s_min_value"%col), "data", minv)
            setattr(getattr(form, "%s_max_value"%col), "data", maxv)
        setattr(getattr(form, "random_seed"), "data", 
                        session.get("random_seed", 1))
        setattr(getattr(form, "target_num_segments"), "data", 
                        session.get("target_num_segments", 12))

    if form.validate_on_submit():
        # filter segments
        session["random_seed"] = form.random_seed.data
        session["target_num_segments"] = form.target_num_segments.data
        for col in sampling_criteria_cols:
            minv = getattr(getattr(form, "%s_min_value"%col), "data")
            session["sampling_%s_min_value"%col] = minv
            dft = dft[dft[col] >= minv]
            maxv = getattr(getattr(form, "%s_max_value"%col), "data")
            session["sampling_%s_max_value"%col] = maxv
            dft = dft[dft[col] <= maxv]
        dft = dft.sample(n=min(form.target_num_segments.data, len(dft)),
                         replace=False, random_state=form.random_seed.data)
        sampled_records = dft.to_dict("records")
        session["sampled_records"] = sampled_records

    return render_template("lenasampler/sampling.html", 
                            form=form,
                            columns=columns,
                            records=sampled_records)


@bp.route("/export_sampled_audio", methods=["GET", "POST"])
def export_sampled_audio():
    sampled_records = session.get('sampled_records', [])
    columns = session.get('columns', [])
    records = session.get('records', [])
    df = pd.DataFrame.from_records(sampled_records)
    df_ori = pd.DataFrame.from_records(records)
    audiodir = session.get('audio_dir', None)
    idprefix = session.get("idprefix", "")
    idprefix = idprefix.strip("_") # remove any trailing _
    itsfilecol = app.config["ITS_FILENAME_COL"]
    starttimecol = app.config["START_TIME_COL"]
    durationcol = app.config["DURATION_COL"]

    form = ExportForm()
    if request.method == "GET":
        form.export_filename.data = "%s_SampledAudioSegments.zip"%idprefix

    if form.validate_on_submit():
        outdir = "%s_SampledAudioSegments_%s"%(idprefix, uuid.uuid4())
        df = prepare_audio_files(df, df_ori, audiodir, outdir, idprefix, 
                                itsfilecol, starttimecol, durationcol)
        # prepare zipfile for download
        zipout = outdir + ".zip"
        with ZipFile(zipout, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipdir(outdir, zipf)
        shutil.rmtree(outdir, ignore_errors=True)
        with open(zipout, 'rb') as f:
            data = f.readlines()
        os.remove(zipout)
        export_fn = form.export_filename.data
        if not export_fn:
            export_fn = "%s_SampledAudioSegments.zip"%idprefix
        if not export_fn.endswith(".zip"):
            export_fn = export_fn + ".zip"

        return Response(data, headers={
            'Content-Type': 'application/zip',
            'Content-Disposition': 'attachment; filename=%s;'%export_fn
        })

    return render_template("lenasampler/export.html", 
                            form=form,
                            columns=columns,
                            records=sampled_records)


@bp.route("/reset_session", methods=["GET", "POST"])
def reset_session():
    session.clear()
    return render_template("lenasampler/reset_session.html")