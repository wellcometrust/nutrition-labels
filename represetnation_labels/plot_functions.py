import json

import numpy as np
import math
from bokeh.plotting import figure, output_file, show, gridplot,save

from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Div, Label, CustomJS, Span
from bokeh.models.widgets import Panel, Tabs
from datetime import date
import re

def missing_plot(source,missing_vals,values):

    missing_val = source.data[missing_vals][0]
    val_sum = sum(source.data[values])
    missing_percent = round(missing_val / (val_sum + missing_val) * 100, 1)

    u = figure(plot_height=50, plot_width=600, toolbar_location=None,x_range=(0,val_sum + missing_val))

    if missing_percent < 0.1:
        text = 'There is no missing data in this variable'
    else:
        text = str(missing_percent) + ' % of data for this variable is missing'

    u.patch(x=[0, 0, val_sum, val_sum], y=[0, 2, 2, 0], color='#eec344', line_color='#c5c5c5', line_width=5)
    u.patch(x=[val_sum, val_sum, val_sum + missing_val, val_sum + missing_val], y=[0, 2, 2, 0], color='#c5c5c5',line_width =5)


    citation = Label(x=5, y=5, x_units='screen', y_units='screen',
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
            legend_name = addition + 'reletive_representative_or_not'
        else:
            legend_name = name + ' percent'

        p = figure(
            x_range=list(source.data['Age']),
            title='Age',
            y_range=(0, max(source.data[values]) * 1.3),
            toolbar_location=None
        )

        if reletivise:

            p.vbar(
                x='Age',
                top=values,
                width=0.9,
                color=addition + 'reletive_colours',
                legend_group=legend_name,
                line_alpha=0,
                source=source
            )
            # You can't add legends to spans so this is the hacky way - create a line with the same settings as the lines, but with no data
            p.line([], [], legend_label='Representative', line_color='black', line_width=1, line_alpha=0.8, line_dash='dashed')

            # Add horizontal line at 100 with label
            hline = Span(location=95, dimension='width', line_color='black', line_width=1, line_alpha=0.8, line_dash='dashed')
            hline2 = Span(location=105, dimension='width', line_color='black', line_width=1, line_alpha=0.8, line_dash='dashed')
            p.renderers.extend([hline, hline2])

            citation = Label(x=30, y=90, x_units='screen', y_units='data',
                 text='95%', render_mode='css')
            p.add_layout(citation)
            citation = Label(x=30, y=105, x_units='screen', y_units='data',
                 text='105%', render_mode='css')
            p.add_layout(citation)

            hover2 = HoverTool(tooltips=[
                ('Age range', '@Age'),
                (' ', "@{" + rel_text + "}"),
            ],
                mode='mouse', name='data plot')
        else:

            p.vbar(
                x='Age',
                top=values,
                width=0.9,
                color='#003667',
                legend_label=legend_name,
                line_alpha=0,
                source=source
            )

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
        p.background_fill_color = '#f2f6fe'
        p.background_fill_alpha = 1
        p.legend.location = 'top_left'
        p.title.text_color = '#5C5C5C'
        p.title.text_font_size = '24pt'
        p.title.text_font = "helvetica"
        p.legend.label_text_font = "helvetica"
        p.legend.label_text_color = "#a0a0a0"
        p.legend.border_line_color = '#555555'
        p.legend.border_line_width = 2
        p.add_tools(hover2)

        a = missing_plot(source, missing_vals, addition + 'values')
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
            x_rep_low_thresh = addition + 'x_l_rep'
            y_rep_low_thresh = addition + 'y_l_rep'
            x_rep_up_thresh = addition + 'x_u_rep'
            y_rep_up_thresh = addition + 'y_u_rep'
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
            tips = addition + 'tips'

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
            text_color='#A65D25',
            source=source,
            render_mode='canvas'
        )

        if reletivise:

            q.patch(
                x= x_vals,
                y=y_vals,
                line_alpha=0,
                color='#003667',
                source=source
            )

            q.circle(
                x= x_vals,
                y= y_vals,
                source=source,
                color=addition + 'reletive_colours',
                size = 6,
                legend_group = addition + 'reletive_representative_or_not')


            q.patch(
                x= x_rep_low_thresh,
                y= y_rep_low_thresh,
                fill_alpha=0,
                line_color='black',
                line_width=1,
                line_alpha=0.8,
                line_dash='dashed',
                source=source
            )

            q.patch(
                x= x_rep_up_thresh,
                y= y_rep_up_thresh,
                fill_alpha=0,
                line_color='black',
                line_width=1,
                line_alpha=0.8,
                line_dash='dashed',
                source=source,
                legend_label='Representative.'
            )

            hover = HoverTool(tooltips=[
                ('Ethnicity', '@Ethnicity'),
                (' ', "@{"+rel_text+"}"),
            ],
                mode='mouse', name='data plot')
        else:

            q.patch(
                x= x_vals,
                y=y_vals,
                line_alpha=0,
                color='#003667',
                source=source,
                legend_label=legend_name
            )

            hover = HoverTool(tooltips=[
                ("Ethnicity", "@Ethnicity"),
                ('Number of people', "@{"+values+"}"),
                ('',"@{"+tips+"}"),
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

        q.multi_line(
            x_lines,
            y_lines,
            source=source,
            color="#a0a0a0",
            line_width=1
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
        q.background_fill_color = '#f2f6fe'
        q.background_fill_alpha = 1
        q.legend.location = 'top_left'
        q.title.text_color = '#5C5C5C'
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

def plot_ethnicity2(source,name,out=False):
    def eth_plot2(source, addition, name):
        y_coords = addition + 'y_coords'
        values = addition + 'values'
        percent = addition + 'percent'
        tips = addition + 'tips'
        missing = addition + 'missing'

        q = figure(title='Ethnicity - Cumulative percent', x_range=(-10, 110))

        q.patches(xs='x_coords', ys=y_coords, color='colours', line_color = 'white',line_width =1.5,legend_field='Ethnicity', source=source)

        perc_lab_cords = np.array([i[0] for i in source.data[y_coords]] + [100])
        perc_x_lab_cords = np.array([0] * len(perc_lab_cords))
        y_labels = [str(i) + '%' for i in perc_lab_cords]
        perc_lab_cords = perc_lab_cords - 0.5

        ref_p_lab_cords = np.array([i[3] for i in source.data[y_coords]] + [100])
        ref_p_x_lab_cords = np.array([100] * len(perc_lab_cords))
        ref_y_labels = [str(i) + '%' for i in ref_p_lab_cords]
        ref_p_lab_cords = ref_p_lab_cords - 0.5

        hover4 = HoverTool(tooltips=[
            ('Ethnicity', '@Ethnicity'),
            ('Number of people', "@{"+values+"}"),
            ('',"@{"+tips+"}"),
            ('Percent/%', "@{"+percent+"}{0.0}"),
            ('UK population percent/%', '@{ref percent}{0.0}')
        ],
            mode='mouse', name='data plot')

        for i in range(len(perc_lab_cords)):
            label = Label(x=perc_x_lab_cords[i], y=perc_lab_cords[i], text=y_labels[i], render_mode='canvas',
                          text_align='right',
                          border_line_alpha=0, background_fill_alpha=0, text_font='helvetica', text_color='#a0a0a0',
                          text_font_size='10pt')

            label2 = Label(
                x=ref_p_x_lab_cords[i],
                y=ref_p_lab_cords[i],
                text=ref_y_labels[i],
                render_mode='canvas',
                text_align='left',
                border_line_alpha=0,
                background_fill_alpha=0,
                text_font='helvetica',
                text_color='#a0a0a0',
                text_font_size='10pt')

            q.add_layout(label)
            q.add_layout(label2)

        dataset_lab = Label(x=20, y=100, text=name, render_mode='canvas', text_align='right',
                            border_line_alpha=0, background_fill_alpha=0, text_font='helvetica', text_color='#a0a0a0')
        ref_lab = Label(x=100, y=100, text='UK Population', render_mode='canvas', text_align='right',
                        border_line_alpha=0, background_fill_alpha=0, text_font='helvetica', text_color='#a0a0a0')

        q.yaxis.major_label_text_font_size = '0pt'
        q.yaxis.major_tick_line_color = None
        q.yaxis.minor_tick_line_color = None
        q.xaxis.major_tick_line_color = None  # turn off y-axis major ticks
        q.xaxis.minor_tick_line_color = None
        q.yaxis.axis_line_color = None
        q.xaxis.axis_line_color = None
        q.xaxis.major_label_text_font_size = '0pt'
        q.xaxis.major_tick_line_color = None
        q.xgrid.grid_line_color = None
        q.ygrid.grid_line_color = None
        q.outline_line_width = 0
        q.background_fill_color = '#f2f6fe'
        q.background_fill_alpha = 1
        q.title.text_color = '#5C5C5C'
        q.title.text_font_size = '24pt'
        q.title.text_font = "helvetica"
        q.legend.location = (46, 24)
        q.legend.label_text_font = "helvetica"
        q.legend.label_text_color = "#a0a0a0"
        q.legend.background_fill_alpha = 0.2
        q.add_layout(dataset_lab)
        q.add_layout(ref_lab)
        q.add_tools(hover4)

        a = missing_plot(source, missing, values)
        text = Div(
            text=source.data['description text'][0],
            style={'font': 'helvetica', 'color': '#555555', 'font-size': '14pt'}
        )
        final_plot = gridplot([[q], [a], [text]], toolbar_options={'autohide': True})

        return final_plot

    if 'values' in source.data.keys():
        eht2_p = eth_plot2(source, '', name)

    else:
        var_list = [re.sub('values', '', i) for i in source.data.keys() if 'values' in i]
        tab_list = [Panel(child=eth_plot2(source, i, name), title=re.sub(' ', '', i)) for i in var_list]
        eht2_p = Tabs(tabs=tab_list)
    if out:
        file_name = str(date.today()) + 'ethnicity_plot.html'
        output_file(file_name)
        save(eht2_p)
    return eht2_p



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
            legend_name = addition + 'reletive_representative_or_not'
        else:
            legend_name = name + ' percent'
        r = figure(
            x_range=list(source.data['Gender']),
            title='Gender',
            y_range=(0, max(source.data[values]) * 1.3),
            toolbar_location=None
        )

        if reletivise:

            r.vbar(
                x='Gender',
                top=values,
                width=0.8,
                color=addition + 'reletive_colours',
                legend_group=legend_name,
                line_alpha=0,
                source=source
            )

            # You can't add legends to spans so this is the hacky way - create a line with the same settings as the lines, but with no data
            r.line([], [], legend_label='Representative', line_color='black', line_width=1, line_alpha=0.8, line_dash='dashed')

            # Add horizontal line at 100 with label
            hline = Span(location=95, dimension='width', line_color='black', line_width=1, line_alpha=0.8, line_dash='dashed')
            hline2 = Span(location=105, dimension='width', line_color='black', line_width=1, line_alpha=0.8, line_dash='dashed')
            r.renderers.extend([hline, hline2])

            citation = Label(x=30, y=90, x_units='screen', y_units='data',
                 text='95%', render_mode='css')
            r.add_layout(citation)
            citation = Label(x=30, y=105, x_units='screen', y_units='data',
                 text='105%', render_mode='css')
            r.add_layout(citation)

            hover2 = HoverTool(tooltips=[
                ('Gender', '@Gender'),
                (' ', "@{" + rel_text + "}"),
            ],
                mode='mouse', name='data plot')
        else:
            r.vbar(
                x='Gender',
                top=values,
                width=0.8,
                color='#003667',
                legend_label=legend_name,
                line_alpha=0,
                source=source
            )

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
        r.background_fill_color = '#f2f6fe'
        r.background_fill_alpha = 1
        r.legend.location = 'top_left'
        r.title.text_color = '#5C5C5C'
        r.title.text_font_size = '24pt'
        r.title.text_font = "helvetica"
        r.legend.label_text_font = "helvetica"
        r.legend.label_text_color = "#a0a0a0"
        r.legend.border_line_color = '#555555'
        r.legend.border_line_width = 2
        r.add_tools(hover2)
        r.toolbar_location = None
        a = missing_plot(source, missing_vals, addition + 'values')
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
            x_rep_low_thresh = addition + 'x_l_rep'
            y_rep_low_thresh = addition + 'y_l_rep'
            x_rep_up_thresh = addition + 'x_u_rep'
            y_rep_up_thresh = addition + 'y_u_rep'
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
            text_color='#A65D25',
            source=source,
            render_mode='canvas'
        )

        if reletivise:

            q.patch(
                x= x_vals,
                y=y_vals,
                line_alpha=0,
                color='#003667',
                source=source
            )

            q.circle(
                x= x_vals,
                y= y_vals,
                source=source,
                color=addition + 'reletive_colours',
                size = 6,
                legend_group = addition + 'reletive_representative_or_not')

            q.patch(
                x= x_rep_low_thresh,
                y= y_rep_low_thresh,
                fill_alpha=0,
                line_color='black',
                line_width=1,
                line_alpha=0.8,
                line_dash='dashed',
                source=source
            )

            q.patch(
                x= x_rep_up_thresh,
                y= y_rep_up_thresh,
                fill_alpha=0,
                line_color='black',
                line_width=1,
                line_alpha=0.8,
                line_dash='dashed',
                source=source,
                legend_label='Representative.'
            )

            hover = HoverTool(tooltips=[
                ('Socioeconomic Status', '@{Socioeconomic Status}'),
                (' ', "@{"+rel_text+"}"),
            ],
                mode='mouse', name='data plot')
        else:

            q.patch(
                x= x_vals,
                y=y_vals,
                line_alpha=0,
                color='#003667',
                source=source,
                legend_label=legend_name
            )

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


        q.multi_line(
            x_lines,
            y_lines,
            source=source,
            color="#a0a0a0",
            line_width=1
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
        q.background_fill_color = '#f2f6fe'
        q.background_fill_alpha = 1
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
        q.title.text_color = '#5C5C5C'
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
    if reletivise:
        eth_p = plot_ethnicity(eth_source, name=name, reletivise=reletivise)
    else:
        eth_p = plot_ethnicity2(eth_source,name = name)
    gender_p = plot_gender(gender_source, name=name, reletivise=reletivise)
    ses_p = plot_ses(ses_source, name=name, reletivise=reletivise)
#    compar_text = Div(text = """<b>'Dataset and UK Population Ratio':</b> tab shows how each demographic group is represented in comaprison to the makeup of the UK. This is calculated by taking the percent of a group in a data set and dividing it by the percent of that group in the UK population. For example: If women make up 10% of a dataset, and women comprise 50% of the population at large, that means this dataset has 20% of the number of women required to be truly representative in this metric (this doesnt include missing data)""",
#                      style={'font-size': '14pt', 'color': '#555555', 'font': 'helvetica'})
    final_plot = gridplot([[age_p,eth_p],[gender_p,ses_p]],toolbar_options={'autohide': True})

    if out:
        if reletivise:
            rel = 'rel'
        else:
            rel = 'no_rel'
        file_name = 'represetnation_labels/web_page/'+ name + '_' + rel + '_' + 'full_plot.html'
        output_file(file_name)
        save(final_plot)
    return final_plot

def rel_plots(plot_dict,name,data_desc_dict,out):
    age_source = ColumnDataSource(data=plot_dict[name]['Age'])
    eth_source = ColumnDataSource(data=plot_dict[name]['Ethnicity'])
    gender_source = ColumnDataSource(data=plot_dict[name]['Gender'])
    ses_source = ColumnDataSource(data=plot_dict[name]['Socioeconomic Status'])
    non_reletive = full_plot(age_source,eth_source,gender_source,ses_source,name,reletivise = False,out = False)
    reletive = full_plot(age_source,eth_source,gender_source,ses_source,name,reletivise = True,out = False)
    pal_nr = Panel(child = non_reletive,title = 'Populations')
    pal_r = Panel(child=reletive, title='Dataset and UK Population Ratio')
    tabs = Tabs(tabs = [pal_nr,pal_r])
    text = data_desc_dict[name]
    data_desc = Div(
        text=text,
        style={'font-size': '14pt', 'color': '#555555', 'font': 'helvetica'}
    )
    out_plot = gridplot([[data_desc],[tabs]],toolbar_options={'autohide': True})
    if out:
        file_name = str(date.today()) + 'tabs_plot.html'
        output_file(file_name)
        save(out_plot)
    return(out_plot)

def all_datasets(plot_dict,data_desc_dict,out):
    panal_list = [Panel(child = rel_plots(plot_dict,i,data_desc_dict,out= False),title = i) for i in plot_dict.keys()]
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
        text="""<ul><li>UK Biobank</li>
                    <li>ALSPAC</li>
                    <li>CPRD</li>
                    <li>National Child Development Study</li>
                    <li>Whitehall study II</li>
                    <li>Hospital Episode Statistics (HES)</li></ul>  """,
        style={'font-size': '10pt', 'color': '#555555', 'font': 'helvetica'})

    creation = Div(
        text="""It was created by the Wellcome Trust with the intention of comparing how the UK population is represented in these datasets, and highlighting where there are disparities.<br>
         <b>How we chose the datasets and accessed the number going into the graph:</b>  The datasets represented by these labels are some of the most commonly used and cited datasets in the UK today. The data displayed here was collated using a combination of metadata available in published papers and online platforms (such as Closer Discovery and the datasets own webpages). No raw data was accessed for the purpose of this project.<br>
         <b>Known limitations:</b> We know that the groupings of sub-populations used in the datasets, e.g. ethnicity groupings, are subjective and potentially inaccurate at times.<br>
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

def export_plots_as_html(plot_dict,reletivise):
    for k in graph_dict2.keys():
        age_source = ColumnDataSource(data=plot_dict[k]['Age'])
        eth_source = ColumnDataSource(data=plot_dict[k]['Ethnicity'])
        gender_source = ColumnDataSource(data=plot_dict[k]['Gender'])
        ses_source = ColumnDataSource(data=plot_dict[k]['Socioeconomic Status'])
        full_plot(age_source, eth_source, gender_source, ses_source, k, reletivise=reletivise,out = True)




if __name__ == '__main__':
    import represetnation_labels.useful_functions as uf

    with open('data/raw/cohort_demographics_test_data.json', 'r') as fb:
        cohorts_dic = json.load(fb)

    with open('data/raw/Reference_population.json', 'r') as fb:
        reference_dict = json.load(fb)

    ref_dict, graph_dict = uf.clean_data(cohorts_dic, reference_dict)
    graph_dict2 = uf.update_graph_dict(graph_dict)
    graph_dict2['UK Biobank']['Text'] = 'The UK Biobank is a prospective cohort study that recruited adults aged between 40-69 years in the UK in 2006-2010. People were invited to participate by mailed invitations to the general public living within 25 miles of one of the 22 assessment centres in England, Scotland and Wales (there was a response rate of 5.5%). '
    graph_dict2['ALSPAC']['Text'] = 'The Avon Longitudinal Study of Children and Parents (ALSPAC) is a prospective cohort study which recruited pregnant women living in the South West of England during 1990-1992. It aims to understand how genetic and environmental factors influence health and development in parents and children by collecting information on demographics, lifestyle behaviours, physical and mental health. The parents and children have been followed up since recruitment through questionnaires, and a subset completed additional assessments (e.g. ‘Focus on Mothers’) which collected anthropometric measurements and biological samples.'
    graph_dict2['ALSPAC']['Age']['description text'] =['At recruitment, the mother was asked to describe her age and that of her partner. The children were obviously all born shortly after their mothers were invited to join the study, so their age at recruitment is 0 years. Overtime, subsequent data was collected at different time points, providing a longitudinal perspective on key health and lifestyle characteristics. So, whilst these labels reflect the baseline characteristics, it does not capture any changes during the participants’ life course (for example when the children are grown-up, their socioeconomic status may be different).'] * len(graph_dict2['ALSPAC']['Age']['Age'])
    graph_dict2['ALSPAC']['Ethnicity']['description text'] = ['The mother was asked to describe the ethnic origin of herself, her partner and her parents in a questionnaire. There were 9 possible ethnicity categories: white, Black/Caribbean, Black/African, Black/other, Indian, Pakistani, Bangladeshi, Chinese, Other. Most research using this data derived the childs ethnic background as ‘white’ (if both parents were described as white) or ‘non-white’ (if either parent was described as any ethnicity other than white). The 9 categories for ethnicity offer a greater level of granularity than many other cohort studies. However, there are far more ethnic groups represented in the UK, and often people do not identify with one ethnicity. These groups also get aggregated into just 2 categories (white or non-white) for the child’s ethnicity, meaning that it may be difficult to understand any nuances or differences in health and well-being related to ethnic background.  Often larger but fewer categories are used for analysis to ensure the sample size is large enough for statistical signifacince.'] * len(graph_dict2['ALSPAC']['Ethnicity']['Ethnicity'])
    for dataset in graph_dict2.keys():
        graph_dict2[dataset]['Age']['description text'] = ['Data shown here reflects the age of participants when they were recruited into a study (for ALSPAC, UK Biobank, Whitehall II and 1958), or their age at the time of hospital or GP episode (HES, CPRD).'] * len(graph_dict2[dataset]['Age']['Age'])
        graph_dict2[dataset]['Ethnicity']['description text'] = ['The 5 ethnicities are the groups which all datasets have in common. Some datasets did collect more granular data (up to 16 categories) but to compare the representativeness between datasets, the data has been grouped to these higher-level categories.']* len(graph_dict2[dataset]['Ethnicity']['Ethnicity'])
        graph_dict2[dataset]['Gender']['description text'] = ['All datasets categorised participants into the sex participants were assigned at birth, either male or female. '] * len(graph_dict2[dataset]['Gender']['Gender'])
        graph_dict2[dataset]['Socioeconomic Status']['description text'] = ['Social class based on Occupation (formerly the UK Registrar General’s occupational coding) has been used across datasets as an indicator of socioeconomic status. The categories are<br><ul><li>V (unskilled)</li><li>IV (semi-skilled manual)</li><li>III (skilled manual)</li><li>III (non-manual)</li><li>II (managerial and technical)</li><li>I (professional)</li></ul>'] * len(graph_dict2[dataset]['Socioeconomic Status']['Socioeconomic Status'])
        if 'values' in graph_dict2[dataset]['Ethnicity'].keys():
            var_list = ['']
        else:
            var_list = [re.sub('values', '', i) for i in graph_dict2[dataset]['Ethnicity'].keys() if 'values' in i]
        boxy_y = {i + 'y_coords': uf.boxy_sanky(graph_dict2[dataset]['Ethnicity'], i) for i in var_list}
        graph_dict2[dataset]['Ethnicity'].update(boxy_y)
        graph_dict2[dataset]['Ethnicity']['x_coords'] = [[0, 0, 100, 100] for i in
                                                         range(len(graph_dict2[dataset]['Ethnicity']['Ethnicity']))]
        graph_dict2[dataset]['Ethnicity']['colours'] = ["#dbeaff","#9dd8e7","#006272","#ffe699","#fec200"]

    for dataset,variables in cohorts_dic.items():
        if dataset in ['UK Biobank','National Child Development Study', 'Whitehall II study', 'HES']:
            graph_dict2[dataset]['Ethnicity']['tips'] = uf.ethnicity_tips(variables['Ethnicity'])
        else:
            tips = {str(k) + ' tips':uf.ethnicity_tips(v) for k,v in variables['Ethnicity'].items()}
            graph_dict2[dataset]['Ethnicity'].update(tips)

    data_desc_dict = {
        'UK Biobank': 'The UK Biobank is a prospective cohort study that recruited 500,000 adults aged between 40-69 years in the UK in 2006-2010, aiming to improve the prevention, diagnosis and treatment of a wide range of illnesses.',
        'ALSPAC': 'The Avon Longitudinal Study of Children and Parents (ALSPAC) is a prospective cohort study which recruited 14,541 pregnant women living in the South West of England during 1990-1992, aiming to understand how genetic and environmental factors influence health and development in parents and children by collecting information on demographics, lifestyle behaviours, physical and mental health',
        'CPRD':'The Clinical Practice Research Datalink (CPRD, formerly the General Practice Research Database) is a primary care research database which collates de-identified patient data from general practices across the UK, covering 50 million patients, including 16 million currently registered patients.',
        'National Child Development Study':'The 1958 National Child Development Study (NCDS) is a prospective cohort study of 17,415 people born in England, Scotland and Wales in a single week of 1958. It has been used to understand topics such as the effects of socioeconomic circumstances, and child adversities on health, and social mobility.',
        'Whitehall II study':'The Whitehall II Study (also know as the Stress and Health Study) is a prospective cohort of 10,308 participants aged 35-55, of whom 3,413 were women and 6,895 men, was recruited from the British Civil Service in 1985.',
        'HES':'Hospital Episode Statistics (HES) is a database containing details of all admissions, A&E attendances and outpatient appointments at NHS hospitals in England. It contains over 200 million records with a wide range of information about an individual patient admitted to an NHS hospital such as diagnoses, operations, demographics, and administrative information. It is often used by linking to other datasets such as the UK Biobank.'
                      }
#    all_datasets(graph_dict2,data_desc_dict, out=True)

    export_plots_as_html(graph_dict2,reletivise=True)
    export_plots_as_html(graph_dict2, reletivise=False)



