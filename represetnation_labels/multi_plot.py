"""
Lots of help from:
https://stackoverflow.com/questions/46831371/how-to-update-plot-by-setting-source-with-select-widget-in-bokeh

Run this in the terminal by:
bokeh serve --show represetnation_labels/multi_plot.py
"""

from bokeh.plotting import figure, output_file, show, output_notebook, gridplot
from bokeh.models.widgets import Select
from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Div
from bokeh.io import curdoc, save
from bokeh.layouts import column, row
from bokeh.client import push_session, pull_session
import numpy as np
import math
import pandas as pd
import os
import json

import useful_functions as uf
from plot_functions import plot_gender, plot_ethnicity, plot_age, plot_ses


with open('data/raw/cohort_demographics_test_data.json', 'r') as fb:
    cohorts_dict = json.load(fb)
with open('data/raw/Reference_population.json', 'r') as fb:
    reference_dict = json.load(fb)

ref_dict, graph_dict = uf.clean_data(cohorts_dict, reference_dict)
graph_dict2 = uf.update_graph_dict(graph_dict)
datasets = list(graph_dict.keys())





# Start with biobank, could also be blank

name = 'UK Biobank'
agesource = ColumnDataSource(data = graph_dict[name]['Age'],name = name)
gendersource = ColumnDataSource(data = graph_dict[name]['Gender'],name = name)
ethsource = ColumnDataSource(data = graph_dict2[name]['Ethnicity'], name = name)
sessource = ColumnDataSource(data = graph_dict2[name]['Socioeconomic Status'], name = name)

age_plot = plot_age(agesource,name)
gender_plot = plot_gender(gendersource,name)

eth_plot = plot_ethnicity(ethsource,name)
soc_plot = plot_ses(sessource,name)

#eth_plot = plot_ethnicity(source)

# gender_plot = figure(x_range = list(source.data['Age - Age']), title = 'Gender', x_axis_label = 'Gender')
# gender_plot.vbar(x = 'Age - Age', top = 'Age - values', width = 0.9, source = source, color= 'blue')


#eth_plot = figure(x_range = list(source.data['Age - Age']), title = 'Gender', x_axis_label = 'Gender')
#eth_plot.vbar(x = 'Age - Age', top = 'Age - values', width = 0.9, source = source, color= 'blue')



p = gridplot([[eth_plot, gender_plot], [age_plot, soc_plot]])

select = Select(title="Select dataset ...",  options=datasets)

def update_plot(attrname, old, new):

    newGenderSource = graph_dict[select.value]['Gender']
    newEthnicitySource = graph_dict2[select.value]['Ethnicity']
    newAgeSource = graph_dict[select.value]['Age']
    newSesSource = graph_dict2[select.value]['Socioeconomic Status']
    gendersource.data = newGenderSource
    ethsource.data = newEthnicitySource
    agesource.data = newAgeSource
    sessource.data = newSesSource
    ethsource.name = select.value
    agesource.name = select.value
    gendersource.name = select.value
    sessource.name = select.value

def change_update_plot(attrname,old,new):
    agesource = ColumnDataSource(data=graph_dict[select.value]['Age'])
    gendersource = ColumnDataSource(data=graph_dict[select.value]['Gender'])
    ethsource = ColumnDataSource(data=graph_dict2[select.value]['Ethnicity'])
    sessource = ColumnDataSource(data=graph_dict2[select.value]['Socioeconomic Status'])
    age_plot = plot_age(agesource, select.value)
    gender_plot = plot_gender(gendersource, select.value)
    eth_plot = plot_ethnicity(ethsource, select.value)
    soc_plot = plot_ses(sessource, select.value)


select.on_change('value', update_plot)
layout = column(row(select, width=400), p)
curdoc().add_root(layout)
