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

with open('data/raw/cohort_demographics_test_data.json', 'r') as fb:
    cohorts_dict = json.load(fb)
with open('data/raw/Reference_population.json', 'r') as fb:
    reference_dict = json.load(fb)

ref_dict, graph_dict = uf.clean_data(cohorts_dict, reference_dict)

datasets = list(graph_dict.keys())

reformatted_dict = {}
for dataset, dataset_dict in graph_dict.items():
    dataset_reformat = {}
    for data_type, values_list in dataset_dict.items():
        for key, value in values_list.items():
            if dataset == 'ALSPAC':
                # Might need a better way for this, but for now only use the Child ALSPAC data
                # The keys also include the generic data e.g. 'Ethnicity' so can't just do
                # 'Child' in key.
                if data_type == 'Socioeconomic Status' or data_type == 'Ethnicity':
                    # Update these to that of the mother (it doesn't exist for the child)
                    if 'Mother' in key:
                        key = key.replace('Mother', 'Child')
                if ('Father' not in key) and ('Mother' not in key):
                    dataset_reformat[data_type + ' - ' + key.replace('Child ','')] = value
            else:
                dataset_reformat[data_type + ' - ' + key] = value
        reformatted_dict[dataset] = dataset_reformat

# Every list needs to be the same length, so pad by adding None's to the end of each
max_length = max([len(v) for v in reformatted_dict['UK Biobank'].values()])
for dataset, dataset_dict in reformatted_dict.items():
    for data_type, values_list in dataset_dict.items():
        # Pad with None's: 
        # e.g. if max_length=5 and values_list=[1,2,3] then -> it becomes [1,2,3,None,None]
        reformatted_dict[dataset][data_type] = (values_list + max_length * [None])[:max_length]

def plot_age(source):

    p = figure(
        x_range = list(source.data['Age - Age']), 
        title = 'Age'
    )

    p.vbar(
        x = 'Age - Age',
        top = 'Age - values', 
        width = 0.9, 
        color = '#003667',
        legend_label = 'UK Biobank',    
        source = source
    )

    p.vbar(
        x = 'Age - Age',
        top = 'Age - reference standardised', 
        width = 0.9, 
        color = '#ffca79', 
        alpha = 0.35,
        line_alpha = 0,
        legend_label = 'UK Population Ratio',
        source = source
    )

    p.line(
        x = 'Age - Age', 
        y = 'Age - missing', 
        line_width = 2, 
        line_color = '#003046',
        legend_label = 'Missing',
        source = source
    )

    hover2 = HoverTool(tooltips = [
        ('Age range', '@{Age - Age}'),
        ('Raw values', '@{Age - values}'),
        ('Percent/%', "@{Age - percent}{0.0}"),
        ('UK population percent/%', '@{Age - ref percent}{0.0}'),
        ('Missing', '@{Age - missing}')
    ],
        mode = 'mouse', name= 'data plot')

    p.yaxis.major_label_text_font_size = '0pt' 
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None
    p.yaxis.axis_line_color = None
    p.xaxis.axis_line_color = None
    p.xaxis.major_label_orientation = math.pi/2
    p.xaxis.major_tick_line_color = 'grey'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.outline_line_width = 0
    p.background_fill_color = '#f5f5f5'
    p.background_fill_alpha = 0.9
    p.legend.location = 'top_left'
    p.title.text_color = '#a0a0a0'
    p.title.text_font_size = '24pt'
    p.title.text_font = "helvetica"
    p.legend.label_text_font = "helvetica"
    p.legend.label_text_color = "#a0a0a0"
    p.add_tools(hover2)

    return p

def plot_ethnicity(source):

    # Number of values, not including the None padding
    ethnicity_values = [i for i in list(source.data['Ethnicity - values']) if ~np.isnan(i)]
    ethnicity_ethnicity = [i for i in list(source.data['Ethnicity - Ethnicity']) if i]
    ethnicity_refstand = [i for i in list(source.data['Ethnicity - reference standardised']) if ~np.isnan(i)]
    ethnicity_missing = [i for i in list(source.data['Ethnicity - missing']) if ~np.isnan(i)]
    ethnicity_percent = [i for i in list(source.data['Ethnicity - percent']) if ~np.isnan(i)]
    ethnicity_refpercent = [i for i in list(source.data['Ethnicity - ref percent']) if ~np.isnan(i)]

    num_vars = len(ethnicity_values)

    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def unit_poly_verts(theta):
        """Return vertices of polygon for subplot axes.
        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
        """
        x0, y0, r = [0.5] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts

    def radar_patch(r, theta):
        yt = (r + 0.01) * np.sin(theta) + 0.5
        xt = (r + 0.01) * np.cos(theta) + 0.5
        return xt, yt

    verts = unit_poly_verts(theta)
    x = [i[0] for i in verts]
    y = [i[1] for i in verts]

    values = np.array(ethnicity_values)
    ref_std = np.array(ethnicity_refstand)
    missing_eth = ethnicity_missing
    missing_eth = missing_eth[0]/sum(values)

    values = values/(sum(values) * 2 )
    ref_std = ref_std/(sum(ref_std) * 2 )

    x_val,y_val = radar_patch(values, theta)
    x_ref,y_ref = radar_patch(ref_std,theta)

    x_lines = [[0.5,i] for i in x]
    y_lines = [[0.5,i] for i in y]

    q = figure(title = 'Ethnicity',x_range=(0,1.08))

    new_source = ColumnDataSource(data=dict(x_vals=x_val,
                                        y_vals=y_val,
                                        x_ref=x_ref,
                                        y_ref=y_ref,
                                        x_lines=x_lines,
                                        y_lines=y_lines,
                                        label_eth = ethnicity_ethnicity,
                                        labs_x_cords = x,
                                        labs_y_cords = y,
                                        values = ethnicity_values,
                                        percent = ethnicity_percent,
                                        ref_perc = ethnicity_refpercent,
                                        missing_list = ethnicity_missing
                                       ))


    labels = LabelSet(
        x='labs_x_cords', 
        y='labs_y_cords', 
        text='label_eth',
        x_offset = 5,
        y_offset =-5, 
        text_font ='helvetica',
        text_color = '#a0a0a0',
        source=new_source,
        render_mode='canvas'
    )

    hover = HoverTool(tooltips=[
            ("Ethnicity", "@label_eth"),
            ('Raw values', '@values'),
            ('Percent/%', "@percent{0.0}"),
            ('UK population percent/%', '@{ref_perc}{0.0}'),
            ('Missing', '@missing_list')
        ])

    q.patch(
        x= 'x_vals', 
        y='y_vals', 
        alpha = 0.35,
        line_alpha = 0,
        color = 'blue',
        source=new_source,
        legend_label = 'UK Biobank'
    )

    q.patch(
        x='x_ref', 
        y='y_ref',
        color = '#ffca79', 
        alpha = 0.35,
        line_alpha = 0,
        source = new_source, 
        legend_label ='UK Population Ratio')

    q.multi_line(
        'x_lines',
        'y_lines',
        source=new_source, 
        color="#a0a0a0", 
        line_width = 1
    )

    q.ellipse(
        x = 0.5,
        y= 0.5,
        width = missing_eth, 
        height = missing_eth, 
        alpha = 0.1, 
        color = '#003046',
        line_alpha = 0,
        legend_label = 'Missing'
    )

    q.yaxis.major_label_text_font_size = '0pt'
    q.xaxis.major_label_text_font_size = '0pt'
    q.yaxis.axis_line_color = None
    q.xaxis.axis_line_color = None
    q.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    q.xaxis.minor_tick_line_color = None
    q.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    q.yaxis.minor_tick_line_color = None
    q.xgrid.grid_line_color = None
    q.ygrid.grid_line_color = None
    q.outline_line_width = 0
    q.background_fill_color = '#f5f5f5'
    q.background_fill_alpha = 0.9
    q.legend.location = 'top_left'
    q.title.text_color = '#a0a0a0'
    q.title.text_font_size = '24pt'
    q.title.text_font = "helvetica"
    q.legend.label_text_font = "helvetica"
    q.legend.label_text_color = "#a0a0a0"
    q.add_layout(labels)
    q.add_tools(hover)

    return q



# Start with biobank, could also be blank
df1 = pd.DataFrame(data=reformatted_dict['UK Biobank'])

source = ColumnDataSource(df1)

age_plot = plot_age(source)

eth_plot = plot_ethnicity(source)

gender_plot = figure(x_range = list(source.data['Age - Age']), title = 'Gender', x_axis_label = 'Gender')
gender_plot.vbar(x = 'Age - Age', top = 'Age - values', width = 0.9, source = source, color= 'blue')

soc_plot = figure(x_range = list(source.data['Age - Age']), title = 'Socioeconomic status', x_axis_label = 'Socioeconomic status')
soc_plot.vbar(x = 'Age - Age', top = 'Age - values', width = 0.9, source = source, color='green')

p = gridplot([[eth_plot, gender_plot], [age_plot, soc_plot]])

select = Select(title="Select dataset ...",  options=datasets)

def update_plot(attrname, old, new):
    newSource = pd.DataFrame(data=reformatted_dict[select.value])
    source.data = newSource

select.on_change('value', update_plot)
layout = column(row(select, width=400), p)
curdoc().add_root(layout)
