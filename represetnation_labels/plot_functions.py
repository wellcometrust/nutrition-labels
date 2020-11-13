import pandas as pd
import json

import numpy as np
import math
from bokeh.plotting import figure, output_file, show, gridplot,save

from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Div, Label, CustomJS
from bokeh.models.widgets import Panel, Tabs
from datetime import date
import re

def missing_plot(source,missing_vals,values):
    u = figure(plot_height = 50,plot_width = 600,toolbar_location = None)
    missing_val = source.data[missing_vals][0]
    val_sum = sum(source.data[values])

    if missing_val == 0:
        text = 'There is no missing data in this variable'
    else:
        missing_percent = round(missing_val/(val_sum + missing_val)*100,1)
        text = 'There is ' + str(missing_percent) + ' % missing data in this variable'


    u.patch(x = [0,0,val_sum,val_sum],y = [0,2,2,0], color = '#ffba79',line_alpha = 0)
    u.patch(x = [val_sum,val_sum,val_sum + missing_val,val_sum + missing_val],y = [0,2,2,0], color = '#c5c5c5',line_alpha = 1)


    citation = Label(x=27, y=5, x_units='screen', y_units='screen',
                     text=text, render_mode='canvas',
                     border_line_alpha=0,
                     background_fill_alpha=0,
                     text_font = 'helvetica',
                     text_color = '#1f1f1f', text_font_size = '19pt'
                    )
    u.add_layout(citation)

    u.yaxis.major_label_text_font_size = '0pt'
    u.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    u.yaxis.minor_tick_line_color = None
    u.yaxis.axis_line_color = None
    u.xaxis.major_label_text_font_size = '0pt'
    u.xaxis.major_tick_line_color = None
    u.xaxis.minor_tick_line_color = None
    u.xaxis.axis_line_color = None
    u.xgrid.grid_line_color = None
    u.ygrid.grid_line_color = None
    u.outline_line_width = 0

    return(u)


def plot_age(source,name,reletivise = False,out = False):
    def age_plot(source, addition, name, reletivise=reletivise):

        if reletivise:
            values = addition + 'reletive'
            rel_text = addition + 'abs text'
        else:
            values = addition + 'values'
            ref_std = addition + 'reference standardised'

        missing_vals = addition + 'missing'
        percent = addition + 'percent'

        if reletivise:
            legend_name = name + '%/Uk Population%'
        else:
            legend_name = name + ' percent'

        p = figure(
            x_range=list(source.data['Age']),
            title='Age',
            y_range=(0, max(source.data[values]) * 1.3),
            toolbar_location=None
        )

        p.vbar(
            x='Age',
            top=values,
            width=0.9,
            color='#003667',
            legend_label=legend_name,
            line_alpha=0,
            source=source
        )

        if reletivise:
            hover2 = HoverTool(tooltips=[
                ('Age range', '@Age'),
                (' ', "@{" + rel_text + "}"),
            ],
                mode='mouse', name='data plot')
        else:
            p.vbar(
                x='Age',
                top=ref_std,
                width=0.8,
                fill_alpha=0,
                line_color='#a0a0a0',
                line_width=4,
                line_alpha=1,
                legend_label='UK Population percent',
                source=source
            )

            hover2 = HoverTool(tooltips=[
                ('Age range', '@Age'),
                ('Number of people', "@{" + values + "}"),
                ('Dataset Percent/%', "@{" + percent + "}{0.0}"),
                ('UK population percent/%', '@{ref percent}{0.0}')
            ],
                mode='mouse', name='data plot')

        p.yaxis.major_label_text_font_size = '0pt'
        p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        p.yaxis.minor_tick_line_color = None
        p.yaxis.axis_line_color = None
        p.xaxis.axis_line_color = None
        p.xaxis.major_label_orientation = math.pi / 2
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
        p.legend.border_line_color = '#555555'
        p.legend.border_line_width = 2
        p.add_tools(hover2)

        a = missing_plot(source, missing_vals, values)
        text = Div(
            text=source.data['description text'][0],
            style={'font': 'helvetica', 'color': '#555555', 'font-size': '14pt'}
        )
        final_plot = gridplot([[p], [a], [text]], toolbar_options={'autohide': True})
        return final_plot

    if 'values' in source.data.keys():
        age_p = age_plot(source, '', name, reletivise=reletivise)
    else:
        var_list = [re.sub('values', '', i) for i in source.data.keys() if 'values' in i]
        tab_list = [Panel(child=age_plot(source, i, name,reletivise=reletivise), title=re.sub(' ', '', i)) for i in var_list]
        age_p = Tabs(tabs=tab_list)
    if out:
        file_name = str(date.today()) + 'age_plot.html'
        output_file(file_name)
        save(age_p)
    return age_p

def plot_ethnicity(source, name,reletivise = False,out = False):

    def ethnicity_plot(source, name,addition,reletivise):
        missing = addition + 'missing'
        if reletivise:
            values = addition + 'reletive'
            rel_text = addition + 'abs text'
            x_vals = addition + 'x_rel'
            y_vals = addition + 'y_rel'
            x_lines = addition + 'rel_x_lines'
            y_lines = addition + 'rel_y_lines'
            labs_x_cords = addition + 'rel_labs_x_cords'
            labs_y_cords = addition + 'rel_labs_y_cords'
        else:
            values = addition + 'values'
            percent = addition + 'percent'
            x_vals = addition + 'x_vals'
            y_vals = addition + 'y_vals'
            x_lines = addition + 'x_lines'
            y_lines = addition + 'y_lines'
            labs_x_cords = addition + 'labs_x_cords'
            labs_y_cords = addition + 'labs_y_cords'
            x_ref = addition + 'x_ref'
            y_ref = addition + 'y_ref'

        if reletivise:
            legend_name = name + '%/Uk Population%'
        else:
            legend_name = name + ' percent'
        q = figure(title='Ethnicity',
                   x_range=(min(source.data[labs_x_cords]) - 0.1, max(source.data[labs_x_cords]) + 0.1),
                   y_range=(min(source.data[labs_y_cords]) * 0.9, max(source.data[labs_y_cords]) * 1.1))

        labels = LabelSet(
            x=labs_x_cords,
            y=labs_y_cords,
            text='Ethnicity',
            text_align='center',
            text_font='helvetica',
            text_color='#a0a0a0',
            source=source,
            render_mode='canvas'
        )

        q.patch(
            x= x_vals,
            y=y_vals,
            line_alpha=0,
            color='#003667',
            source=source,
            legend_label=legend_name
        )

        q.multi_line(
            x_lines,
            y_lines,
            source=source,
            color="#a0a0a0",
            line_width=1
        )
        if reletivise:
            hover = HoverTool(tooltips=[
                ('Ethnicity', '@Ethnicity'),
                (' ', "@{"+rel_text+"}"),
            ],
                mode='mouse', name='data plot')
        else:
            hover = HoverTool(tooltips=[
                ("Ethnicity", "@Ethnicity"),
                ('Number of people', "@{"+values+"}"),
                ('Dataset percent/%', "@{"+percent+"}{0.0}"),
                ('UK population percent/%', '@{ref percent}{0.0}')
            ])

            q.patch(
                x=x_ref,
                y=y_ref,
                color='#a0a0a0',
                line_width=0,
                alpha=0.35,
                source=source,
                legend_label='UK Population Percent')

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
        q.legend.border_line_color = '#555555'
        q.legend.border_line_width = 2
        q.add_layout(labels)
        q.add_tools(hover)
        q.toolbar_location = None
        b = missing_plot(source, missing, addition +'values')
        text = Div(
            text=source.data['description text'][0],
            style={'font': 'helvetica', 'color': '#555555', 'font-size': '14pt'}
        )
        final_plot = gridplot([[q], [b], [text]], toolbar_options={'autohide': True})

        return final_plot


    if 'values' in source.data.keys():
        eth_p = ethnicity_plot(source,  name,'', reletivise=reletivise)
    else:
        var_list = [re.sub('values', '', i) for i in source.data.keys() if 'values' in i]
        tab_list = [Panel(child=ethnicity_plot(source, name, i,reletivise), title=re.sub(' ', '', i)) for i in var_list]
        eth_p = Tabs(tabs=tab_list)
    if out:
        file_name = str(date.today()) + 'eth_plot.html'
        output_file(file_name)
        save(eth_p)
    return eth_p

def plot_gender(source,name, reletivise = False,out = False):
    def gender_plot(source, addition, name, reletivise=reletivise):
        if reletivise:
            values = addition + 'reletive'
            rel_text = addition + 'abs text'
        else:
            values = addition + 'values'
            ref_std = addition + 'reference standardised'

        missing_vals = addition + 'missing'
        percent = addition + 'percent'
        if reletivise:
            legend_name = name + '%/Uk Population%'
        else:
            legend_name = name + ' percent'
        r = figure(
            x_range=list(source.data['Gender']),
            title='Gender',
            y_range=(0, max(source.data[values]) * 1.3),
            toolbar_location=None
        )

        r.vbar(
            x='Gender',
            top=values,
            width=0.8,
            color='#003667',
            legend_label=legend_name,
            line_alpha=0,
            source=source
        )

        if reletivise:
            hover2 = HoverTool(tooltips=[
                ('Gender', '@Gender'),
                (' ', "@{" + rel_text + "}"),
            ],
                mode='mouse', name='data plot')
        else:
            r.vbar(
                x='Gender',
                top=ref_std,
                width=0.8,
                fill_alpha=0,
                line_color='#a0a0a0',
                line_width=4,
                line_alpha=1,
                legend_label='UK Population Percent',
                source=source
            )

            hover2 = HoverTool(tooltips=[
                ('Gender', '@Gender'),
                ('Number of people', "@{" + values + "}"),
                ('Dataset percent/%', "@{" + percent + "}{0.0}"),
                ('UK population percent/%', '@{ref percent}{0.0}')
            ],
                mode='mouse', name='data plot')

        r.yaxis.major_label_text_font_size = '0pt'
        r.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        r.yaxis.minor_tick_line_color = None
        r.yaxis.axis_line_color = None
        r.xaxis.axis_line_color = None
        r.xaxis.major_tick_line_color = 'grey'
        r.xgrid.grid_line_color = None
        r.ygrid.grid_line_color = None
        r.outline_line_width = 0
        r.background_fill_color = '#f5f5f5'
        r.background_fill_alpha = 0.9
        r.legend.location = 'top_left'
        r.title.text_color = '#a0a0a0'
        r.title.text_font_size = '24pt'
        r.title.text_font = "helvetica"
        r.legend.label_text_font = "helvetica"
        r.legend.label_text_color = "#a0a0a0"
        r.legend.border_line_color = '#555555'
        r.legend.border_line_width = 2
        r.add_tools(hover2)
        r.toolbar_location = None
        a = missing_plot(source, missing_vals, values)
        text = Div(
            text=source.data['description text'][0],
            style={'font': 'helvetica', 'color': '#555555', 'font-size': '14pt'}
        )
        final_plot = gridplot([[r], [a], [text]], toolbar_options={'autohide': True})

        return (final_plot)


    if 'values' in source.data.keys():
        gender_p = gender_plot(source, '', name, reletivise=reletivise)

    else:
        var_list = [re.sub('values', '', i) for i in source.data.keys() if 'values' in i]
        tab_list = [Panel(child=gender_plot(source, i, name,reletivise=reletivise), title=re.sub(' ', '', i)) for i in var_list]
        gender_p = Tabs(tabs=tab_list)
    if out:
        file_name = str(date.today()) + 'gender_plot.html'
        output_file(file_name)
        save(gender_p)
    return gender_p

def plot_ses(source, name,reletivise = False,out = False):

    def ses_plot(source, name,addition,reletivise):
        missing = addition + 'missing'
        if reletivise:
            values = addition + 'reletive'
            rel_text = addition + 'abs text'
            x_vals = addition + 'x_rel'
            y_vals = addition + 'y_rel'
            x_lines = addition + 'rel_x_lines'
            y_lines = addition + 'rel_y_lines'
            labs_x_cords = addition + 'rel_labs_x_cords'
            labs_y_cords = addition + 'rel_labs_y_cords'
        else:
            values = addition + 'values'
            percent = addition + 'percent'
            x_vals = addition + 'x_vals'
            y_vals = addition + 'y_vals'
            x_lines = addition + 'x_lines'
            y_lines = addition + 'y_lines'
            labs_x_cords = addition + 'labs_x_cords'
            labs_y_cords = addition + 'labs_y_cords'
            x_ref = addition + 'x_ref'
            y_ref = addition + 'y_ref'

        if reletivise:
            legend_name = name + '%/Uk Population%'
        else:
            legend_name = name + ' percent'

        q = figure(title='Socioeconomic status', x_range=(min(source.data[labs_x_cords]) - 0.1, max(source.data[labs_x_cords]) + 0.1),
                   y_range=(min(source.data[labs_y_cords]) * 0.9, max(source.data[labs_y_cords]) * 1.1))

        labels = LabelSet(
            x=labs_x_cords,
            y=labs_y_cords,
            text='Socioeconomic Status',
            text_align='center',
            text_font='helvetica',
            text_color='#a0a0a0',
            source=source,
            render_mode='canvas'
        )

        q.patch(
            x= x_vals,
            y=y_vals,
            line_alpha=0,
            color='#003667',
            source=source,
            legend_label=legend_name
        )

        q.multi_line(
            x_lines,
            y_lines,
            source=source,
            color="#a0a0a0",
            line_width=1
        )
        if reletivise:
            hover = HoverTool(tooltips=[
                ('Socioeconomic Status', '@{Socioeconomic Status}'),
                (' ', "@{"+rel_text+"}"),
            ],
                mode='mouse', name='data plot')
        else:
            hover = HoverTool(tooltips=[
                ('Socioeconomic Status', '@{Socioeconomic Status}'),
                ('Number of people', "@{"+values+"}"),
                ('Dataset percent/%', "@{"+percent+"}{0.0}"),
                ('UK population percent/%', '@{ref percent}{0.0}')
            ])

            q.patch(
                x=x_ref,
                y=y_ref,
                color='#a0a0a0',
                line_width=0,
                alpha=0.35,
                source=source,
                legend_label='UK Population Percent')

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
        q.legend.border_line_color = '#555555'
        q.legend.border_line_width = 2
        q.add_layout(labels)
        q.add_tools(hover)
        q.toolbar_location = None
        b = missing_plot(source, missing, addition +'values')
        text = Div(
            text=source.data['description text'][0],
            style={'font': 'helvetica', 'color': '#555555', 'font-size': '14pt'}
        )
        final_plot = gridplot([[q], [b], [text]], toolbar_options={'autohide': True})

        return final_plot


    if 'values' in source.data.keys():
        ses_p = ses_plot(source, name,'', reletivise=reletivise)
    else:
        var_list = [re.sub('values', '', i) for i in source.data.keys() if 'values' in i]
        tab_list = [Panel(child=ses_plot(source, name, i, reletivise=reletivise),
                        title=re.sub(' ', '', i)) for i in var_list]
        ses_p = Tabs(tabs=tab_list)
    if out:
        file_name = str(date.today()) + 'ses_plot.html'
        output_file(file_name)
        save(ses_p)
    return ses_p

def full_plot_ethnicity(data_dict,name,reletivise = False, out = False):
    def ethnicity_plot(data_dict, addition, name, reletivise):

        imp_keys = ['Ethnicity', 'ref percent', 'description text'] + [k for k in data_dict.keys() if addition in k]
        test_spider = {re.sub(addition, '', k): v for k, v in data_dict.items() if k in imp_keys}

        num_vars = len(test_spider['values'])
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        # rotate theta such that the first axis is at the top
        theta += np.pi / 2

        def unit_poly_verts(theta):
            """Return vertices of polygon for subplot axes.
            This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
            """
            x0, y0, r = [0.5] * 3
            verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
            return verts

        def radar_patch(r, theta):
            yt = (r + 0.01) * np.sin(theta) + 0.5
            xt = (r + 0.01) * np.cos(theta) + 0.5
            return xt, yt

        verts = unit_poly_verts(theta)
        x = [i[0] for i in verts]
        y = [i[1] for i in verts]

        if reletivise:
            values = np.array(test_spider['reletive'])
        else:
            values = np.array(test_spider['values'])

        ref_std = np.array(test_spider['reference standardised'])

        values = values / (sum(values) * 2)
        ref_std = ref_std / (sum(ref_std) * 2)

        x_val, y_val = radar_patch(values, theta)
        x_ref, y_ref = radar_patch(ref_std, theta)

        label_eth = test_spider['Ethnicity']

        new_line_max = np.array([max(np.concatenate([values, ref_std]))] * len(values))
        new_x, new_y = radar_patch(new_line_max, theta)
        new_x_lines = [[0.5, i] for i in new_x]
        new_y_lines = [[0.5, i] for i in new_y]


        title_eth = 'Ethnicity'

        source = ColumnDataSource(data=dict(x_vals=x_val,
                                            y_vals=y_val,
                                            x_ref=x_ref,
                                            y_ref=y_ref,
                                            x_lines=new_x_lines,
                                            y_lines=new_y_lines,
                                            label_eth=label_eth,
                                            labs_x_cords=new_x,
                                            labs_y_cords=new_y,
                                            values=test_spider['values'],
                                            percent=test_spider['percent'],
                                            ref_perc=test_spider['ref percent'],
                                            missing=test_spider['missing'],
                                            desc_text=test_spider['description text'],
                                            rel_text=test_spider['reletive text']
                                            ))
        q = figure(title=title_eth, x_range=(min(new_x) - 0.1, max(new_x) + 0.1),
                   y_range=(min(new_y) * 0.9, max(new_y) * 1.1))

        labels = LabelSet(
            x='labs_x_cords',
            y='labs_y_cords',
            text='label_eth',
            text_align='center',
            text_font='helvetica',
            text_color='#a0a0a0',
            source=source,
            render_mode='canvas'
        )

        q.patch(
            x='x_vals',
            y='y_vals',
            line_alpha=0,
            color='#003667',
            source=source,
            legend_label=name
        )

        q.multi_line(
            'x_lines',
            'y_lines',
            source=source,
            color="#a0a0a0",
            line_width=1
        )
        if reletivise:
            hover = HoverTool(tooltips=[
                ('Ethnicity', '@label_eth'),
                (' ', "@rel_text"),
            ],
                mode='mouse', name='data plot')
        else:
            hover = HoverTool(tooltips=[
                ("Ethnicity", "@label_eth"),
                ('Raw values', '@values'),
                ('Percent/%', "@percent{0.0}"),
                ('UK population percent/%', '@{ref_perc}{0.0}')
            ])

            q.patch(
                x='x_ref',
                y='y_ref',
                color='#a0a0a0',
                line_width=0,
                alpha=0.35,
                source=source,
                legend_label='UK Population Ratio')

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
        q.toolbar_location = None
        b = missing_plot(source, 'missing', 'values')
        text = Div(
            text=source.data['desc_text'][0],
            style={'font': 'helvetica', 'color': '#555555', 'font-size': '14pt'}
        )
        final_plot = gridplot([[q], [b], [text]], toolbar_options={'autohide': True})

        return (final_plot)

    if 'values' in data_dict.keys():
        eth_p = ethnicity_plot(data_dict, '', name, reletivise=reletivise)
    else:
        var_list = [re.sub('values', '', i) for i in data_dict if 'values' in i]
        tab_list = [Panel(child=ethnicity_plot(data_dict, i, name,reletivise=reletivise), title=re.sub(' ', '', i)) for i in var_list]
        eth_p = Tabs(tabs=tab_list)
    if out:
        file_name = str(date.today()) + 'ethnicity_plot.html'
        output_file(file_name)
        save(eth_p)
    return eth_p

def full_ses_plot(data_dict,name,reletivise = False, out = False):
    def ses_plot(data_dict, addition, name, reletivise):
        imp_keys = ['Socioeconomic Status', 'ref percent', 'description text'] + [k for k in data_dict.keys() if
                                                                                  addition in k]

        test_spider = {re.sub(addition, '', k): v for k, v in data_dict.items() if k in imp_keys}

        num_vars = len(test_spider['values'])

        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        # rotate theta such that the first axis is at the top
        theta += np.pi / 2
        if reletivise:
            legend_name = name + '%/Uk Population%'
        else:
            legend_name = name + ' percent'
        def unit_poly_verts(theta):
            """Return vertices of polygon for subplot axes.
            This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
            """
            x0, y0, r = [0.5] * 3
            verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
            return verts

        def radar_patch(r, theta):
            yt = (r + 0.01) * np.sin(theta) + 0.5
            xt = (r + 0.01) * np.cos(theta) + 0.5
            return xt, yt

        verts = unit_poly_verts(theta)
        x = [i[0] for i in verts]
        y = [i[1] for i in verts]

        if reletivise:
            values = np.array(test_spider['reletive'])
        else:
            values = np.array(test_spider['values'])

        ref_std = np.array(test_spider['reference standardised'])
        missing_ses = test_spider['missing']
        missing_ses = missing_ses[0] / sum(values)

        values = values / (sum(values) * 2)
        ref_std = ref_std / (sum(ref_std) * 2)

        x_val, y_val = radar_patch(values, theta)
        x_ref, y_ref = radar_patch(ref_std, theta)

        label_ses = test_spider['Socioeconomic Status']

        x_lines = [[0.5, i] for i in x]
        y_lines = [[0.5, i] for i in y]
        new_line_max = np.array([max(np.concatenate([values, ref_std]))] * len(values))
        new_x, new_y = radar_patch(new_line_max, theta)
        new_x_lines = [[0.5, i] for i in new_x]
        new_y_lines = [[0.5, i] for i in new_y]

        s = figure(title='Socioeconomic Status', x_range=(min(new_x) - 0.05, max(new_x) + 0.05),
                   y_range=(min(new_y) * 0.9, max(new_y) * 1.1))

        source = ColumnDataSource(data=dict(x_vals=x_val,
                                            y_vals=y_val,
                                            x_ref=x_ref,
                                            y_ref=y_ref,
                                            x_lines=new_x_lines,
                                            y_lines=new_y_lines,
                                            label_ses=label_ses,
                                            labs_x_cords=new_x,
                                            labs_y_cords=new_y,
                                            values=test_spider['values'],
                                            percent=test_spider['percent'],
                                            ref_perc=test_spider['ref percent'],
                                            missing=test_spider['missing'],
                                            desc_text=test_spider['description text'],
                                            rel_text=test_spider['reletive text']
                                            ))

        labels = LabelSet(
            x='labs_x_cords',
            y='labs_y_cords',
            text='label_ses',
            text_font='helvetica',
            text_color='#a0a0a0',
            source=source,
            render_mode='canvas',
            text_align='center'
        )

        s.patch(
            x='x_vals',
            y='y_vals',
            alpha=1,
            line_alpha=0,
            color='#003667',
            source=source,
            legend_label=legend_name
        )
        if reletivise:
            hover = HoverTool(tooltips=[
                ('Socioeconomic Status', '@label_ses'),
                (' ', "@rel_text"),
            ],
                mode='mouse', name='data plot')
        else:
            s.patch(
                x='x_ref',
                y='y_ref',
                color='#a0a0a0',
                alpha=0.35,
                line_alpha=0,
                source=source,
                legend_label='UK Population Percent')

            hover = HoverTool(tooltips=[
                ("Socioeconomic", "@label_ses"),
                ('Raw values', '@values'),
                ('Percent/%', "@percent{0.0}"),
                ('UK population percent/%', '@{ref_perc}{0.0}')
            ])

        s.multi_line(
            'x_lines',
            'y_lines',
            source=source,
            color="#a0a0a0",
            line_width=1
        )

        s.yaxis.major_label_text_font_size = '0pt'
        s.xaxis.major_label_text_font_size = '0pt'
        s.yaxis.axis_line_color = None
        s.xaxis.axis_line_color = None
        s.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
        s.xaxis.minor_tick_line_color = None
        s.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        s.yaxis.minor_tick_line_color = None
        s.xgrid.grid_line_color = None
        s.ygrid.grid_line_color = None
        s.outline_line_width = 0
        s.background_fill_color = '#f5f5f5'
        s.background_fill_alpha = 0.9
        s.legend.location = 'top_left'
        s.title.text_color = '#a0a0a0'
        s.title.text_font_size = '24pt'
        s.title.text_font = "helvetica"
        s.legend.label_text_font = "helvetica"
        s.legend.label_text_color = "#a0a0a0"
        s.add_layout(labels)
        s.add_tools(hover)
        s.toolbar_location = None
        b = missing_plot(source, 'missing', 'values')
        text = Div(
            text=source.data['desc_text'][0],
            style={'font': 'helvetica', 'color': '#555555', 'font-size': '14pt'}
        )
        final_plot = gridplot([[s], [b], [text]], toolbar_options={'autohide': True})
        return (final_plot)


    if 'values' in data_dict.keys():
        ses_p = ses_plot(data_dict, '', name, reletivise=reletivise)
    else:
        var_list = [re.sub('values', '', i) for i in data_dict.keys() if 'values' in i]
        tab_list = [Panel(child=ses_plot(data_dict, i, name,reletivise=reletivise), title=re.sub(' ', '', i)) for i in var_list]
        ses_p = Tabs(tabs=tab_list)
    if out:
        file_name = str(date.today()) + 'socio_plot.html'
        output_file(file_name)
        save(ses_p)
    return(ses_p)

def full_plot(age_source,eth_source,gender_source,ses_source, name,reletivise,out):
    age_p = plot_age(age_source,name = name,reletivise=reletivise)
    eth_p = plot_ethnicity(eth_source, name=name, reletivise=reletivise)
    gender_p = plot_gender(gender_source, name=name, reletivise=reletivise)
    ses_p = plot_ses(ses_source, name=name, reletivise=reletivise)
    final_plot = gridplot([[age_p,eth_p],[gender_p,ses_p]],toolbar_options={'autohide': True})
    if out:
        file_name = str(date.today()) + 'full_plot.html'
        output_file(file_name)
        save(final_plot)
    return final_plot

def rel_plots(plot_dict,name,out):
    age_source = ColumnDataSource(data=plot_dict[name]['Age'])
    eth_source = ColumnDataSource(data=plot_dict[name]['Ethnicity'])
    gender_source = ColumnDataSource(data=plot_dict[name]['Gender'])
    ses_source = ColumnDataSource(data=plot_dict[name]['Socioeconomic Status'])
    non_reletive = full_plot(age_source,eth_source,gender_source,ses_source,name,reletivise = False,out = False)
    reletive = full_plot(age_source,eth_source,gender_source,ses_source,name,reletivise = True,out = False)
    pal_nr = Panel(child = non_reletive,title = 'Populations')
    pal_r = Panel(child=reletive, title='Compare Populations')
    tabs = Tabs(tabs = [pal_nr,pal_r])
    if out:
        file_name = str(date.today()) + 'tabs_plot.html'
        output_file(file_name)
        save(tabs)
    return(tabs)

def all_datasets(plot_dict,out):
    panal_list = [Panel(child = rel_plots(plot_dict,i,out= False),title = i) for i in plot_dict.keys()]
    tabs = Tabs(tabs = panal_list)
    title = Div(text="""Data Representation Labels""",
                style={'font-size': '28pt', 'color': '#a0a0a0', 'font': 'helvetica'}
                )
    created_by = Div(
        text="""<i>Created by Wellcome’s Data for Science and Health Priority Area </i>""",
        style={'font-size': '7pt', 'color': '#555555', 'font': 'helvetica'}
    )
    description  = Div(
        text="""This is a tool for researchers to examine the demographics of commonly used datasets and how they compare to the UK population. The datasets we are highlighting are: """,
        style={'font-size': '14pt', 'color': '#555555', 'font': 'helvetica'}
    )

    dataset_list = Div(
        text="""<ul><li><b>UK Biobank:</b> The UK Biobank is a prospective cohort study that recruited 500,000 adults aged between 40-69 years in the UK in 2006-2010, aiming to improve the prevention, diagnosis and treatment of a wide range of illnesses.</li>
                    <li><b>ALSPAC:</b> The Avon Longitudinal Study of Children and Parents (ALSPAC) is a prospective cohort study which recruited 14,541 pregnant women living in the South West of England during 1990-1992, aiming to understand how genetic and environmental factors influence health and development in parents and children by collecting information on demographics, lifestyle behaviours, physical and mental health</li>
                    <li><b>CPRD:</b> The Clinical Practice Research Datalink (CPRD, formerly the General Practice Research Database) is a primary care research database which collates de-identified patient data from general practices across the UK, covering 50 million patients, including 16 million currently registered patients.</li>
                    <li><b>National Child Development Study:</b> The 1958 National Child Development Study (NCDS) is a prospective cohort study of 17,415 people born in England, Scotland and Wales in a single week of 1958.  It has been used to understand topics such as the effects of socioeconomic circumstances, and child adversities on health, and social mobility.</li>
                    <li><b>Whitehall study II:</b> The Whitehall II Study (also know as the Stress and Health Study) is a prospective cohort of 10,308 participants aged 35-55, of whom 3,413 were women and 6,895 men, was recruited from the British Civil Service in 1985.</li>
                    <li><b>HES:</b> Hospital Episode Statistics (HES) is a database containing details of all admissions, A&E attendances and outpatient appointments at NHS hospitals in England. It contains over 200 million records with a wide range of information about an individual patient admitted to an NHS hospital such as diagnoses, operations, demographics, and administrative information. It is often used by linking to other datasets such as the UK Biobank. </li></ul>  """,
        style={'font-size': '10pt', 'color': '#555555', 'font': 'helvetica'})

    creation = Div(
        text="""It was created by the Wellcome Trust with the intention of comparing how the UK population is represented in these datasets, and highlighting where there are disparities.<br>
         <b>How we chose the datasets and accessed the number going into the graph:</b>  The datasets represented by these labels are some of the most commonly used and cited datasets in the UK today. The data displayed here was collated using a combination of metadata available in published papers and online platforms (such as Closer Discovery and the datasets own webpages). No raw data was accessed for the purpose of this project.<br>
         <b>Known limitations:</b> We know that the groupings of sub-populations used in the datasets, e.g. ethnicity groupings, are subjective and potentially inaccurate at times.<br>
         <b>'Compare Populations':</b> tab shows how each demographic group is represented in comaprison to the makeup of the UK. This is calculated by taking the percent of a group in a data set and dividing it by the percent of that group in the UK population. For example: If women make up 10% of a dataset, and women comprise 50% of the population at large, that means this dataset has 20% of the number of women required to be truly representative in this metric (this doesnt include missing data).<br>
         Please don’t hesitate to contact us with any questions, feedback or suggestions at <u>b.knowles@wellcome.org</ul> """,
        style={'font-size': '14pt', 'color': '#555555', 'font': 'helvetica'}
    )

    last_updated = Div(
        text="""<i>Date last updated: 9th Nov 2020</i> """,
        style={'font-size': '7pt', 'color': '#555555', 'font': 'helvetica'}

    )
    final = gridplot([[title],[created_by],[description], [dataset_list],[creation],[last_updated] ,[tabs]],
                     toolbar_options={'autohide': True})

    if out:
        file_name = str(date.today()) + 'representation_labels.html'
        output_file(file_name)
        save(final)
    return(final)



if __name__ == '__main__':
    all_datasets(graph_dict2, out=True)

