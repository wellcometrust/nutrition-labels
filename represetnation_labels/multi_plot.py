"""
Lots of help from:
https://stackoverflow.com/questions/46831371/how-to-update-plot-by-setting-source-with-select-widget-in-bokeh

Run this in the terminal by:
bokeh serve --show represetnation_labels/multi_plot.py
"""

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show, output_notebook, gridplot
from bokeh.models.widgets import Select
from bokeh.io import curdoc, save
from bokeh.layouts import column, row
from bokeh.client import push_session, pull_session

import pandas as pd
import os
import json

with open('data/raw/cohort_demographics_test_data.json', 'r') as fb:
    cohorts_dic = json.load(fb)
with open('data/raw/Reference_population.json', 'r') as fb:
    ref_dic = json.load(fb)

datasets = list(cohorts_dic.keys())

# UPDATE: reformatted_dict needs to be updated to have the correct x and y 
# lists for each plot and for each dataset, currently just for age

reformatted_dict = {}
for dataset, values in cohorts_dic.items():
    reformatted_dict[dataset] = {
        'Age ranges': list(values['Age'].keys()),
        'Age ranges values': list(values['Age'].values())
    }
    # if dataset=='ALSPAC':
        # Age data has further breakdown

# Start with biobank, could also be blank
df1 = pd.DataFrame(data=reformatted_dict['UK Biobank'])

source = ColumnDataSource(df1)

## UPDATE: PLOTS FOR THE NON-AGE ONES

age_plot = figure(x_range = list(source.data['Age ranges']), title = 'Age', x_axis_label = 'Age ranges')
age_plot.vbar(x = 'Age ranges', top = 'Age ranges values', width = 0.9, source = source, color='red')

gender_plot = figure(x_range = list(source.data['Age ranges']), title = 'Gender', x_axis_label = 'Gender')
gender_plot.vbar(x = 'Age ranges', top = 'Age ranges values', width = 0.9, source = source, color= 'blue')

eth_plot = figure(x_range = list(source.data['Age ranges']), title = 'Ethnicity', x_axis_label = 'Ethnicity')
eth_plot.vbar(x = 'Age ranges', top = 'Age ranges values', width = 0.9, source = source, color='yellow')

soc_plot = figure(x_range = list(source.data['Age ranges']), title = 'Socioeconomic status', x_axis_label = 'Socioeconomic status')
soc_plot.vbar(x = 'Age ranges', top = 'Age ranges values', width = 0.9, source = source, color='green')

p = gridplot([[age_plot, gender_plot], [eth_plot, soc_plot]])

select = Select(title="Select dataset ...",  options=datasets)

def update_plot(attrname, old, new):
    newSource = pd.DataFrame(data=reformatted_dict[select.value])
    source.data = newSource

select.on_change('value', update_plot)
layout = column(row(select, width=400), p)
curdoc().add_root(layout)
