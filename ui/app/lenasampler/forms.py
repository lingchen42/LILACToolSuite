from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField,\
                    SelectMultipleField, FormField, FieldList, IntegerField,\
                    BooleanField
from flask_wtf.file import FileField, FileAllowed
from wtforms.validators import DataRequired, NumberRange
from app import app


class DataInput(FlaskForm):
    fn = FileField("Upload a exported LENA csv file",
                    validators=[FileAllowed(['csv'],
                                 "Only csv file is accepted")])
    audio_dir = StringField("LENA Audio File Folder (contains .wav audio files)") 
    submit = SubmitField('Upload')


class FilterForm(FlaskForm):
    itsfiles = SelectMultipleField("Which files to keep?",
                        default="", choices=[])
    
    def __init__(self, itsfiles=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(itsfiles):
            self.itsfiles.choices = itsfiles

for col in app.config["DEFAULT_FILTER_NUM_COLUMNS"]:
    setattr(FilterForm, "%s_min_value"%col, IntegerField("%s min value"%col))
    setattr(FilterForm, "%s_max_value"%col, IntegerField("%s max value"%col))

setattr(FilterForm, "submit", SubmitField('Confirm'))


class SamplingColsForm(FlaskForm):
    samplingcols \
        = SelectMultipleField("Which columns to be used as sampling criteria?",
                              choices=[])
    submit = SubmitField('Next')
    
    def __init__(self, cols, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(cols):
            self.samplingcols.choices = cols


class ExportForm(FlaskForm):
    export_filename = StringField("Export as")
    submit = SubmitField('Export')