import json
import re
import numpy as np

def change_format(og_dict,name):
    new_dict = {name: list(og_dict.keys()),
                'values': list(og_dict.values())}
    return new_dict

def eth_change_format(eth_dict,name):
    eth_dict_alter = eth_dict.copy()
    eth_dict_alter['Asian']['Chinese'] = eth_dict_alter['Chinese']['Chinese']
    del eth_dict_alter['Chinese']
    new_dict = {name:list(eth_dict_alter.keys()),
                'values':[sum(v.values()) for v in eth_dict_alter.values()]}
    return new_dict

def multi_eth_change_format(eth_dict,dataset):
    if dataset == 'ALSPAC' or dataset == 'CPRD':
        dict_copy = {k: eth_change_format(v, k) for k, v in eth_dict.items()}
        dict_copy = {str(k) + ' values': v['values'] for k, v in dict_copy.items()}
        dict_copy['Ethnicity'] = ['White', 'Black', 'Asian', 'Mixed', 'Other', 'Missing']
        return dict_copy
    else:
        return eth_change_format(eth_dict, 'Ethnicity')

def nested_change_format(nested_dict,name):
    fst_key = list(nested_dict.keys())[0]
    variable = list(nested_dict[fst_key].keys())
    new_dict = {str(k) + ' values':[values for values in v.values()] for k,v in nested_dict.items()}
    new_dict[name] = variable
    return new_dict

def get_percents(vals):
    percents = [round(i/sum(vals[:-1])*100,1) for i in vals[:-1]]
    return percents

def get_reletives(perc,refs_perc):
    reletives = [round(perc[i]/refs_perc[i]*100,1) if refs_perc[i] > 1 else 0 for i in range(len(perc))]
    return reletives

def standardise_refs(vals,ref_perc):
    std = [round(sum(vals[:-1])*i/100) for i in ref_perc]
    return std

def get_name(name_key,addition):
    if name_key == 'values' or name_key == 'percent' or name_key == '':
        return addition
    else:
        alter_name = re.sub(' values','',name_key)
        alter_name = re.sub(' percent','', alter_name)
        alter_name = re.sub('','', alter_name)
        return str(alter_name) + ' ' + (addition)

def reletivise_text(perc):
    string =  'there is ' + str(round(perc)) + '% of this group needed to be representative'
    return string

def absolute_reletivise_text(values,ref_std):
    if values == 0:
        return('This group is representative.')
    elif ref_std > values:
        val = ref_std - values
        mult = round(ref_std/values,1)
        return('This group needs ' + str(mult) + 'X (n:' + str(val) +') more people to be representative.')
    else:
        val = values - ref_std
        return ('This group is overrepresented. There should be ' + str(val) + ' fewer people in this group to be representative.')

def absolute_reletivie_list(values,ref_std):
    return([absolute_reletivise_text(values[i],ref_std[i]) for i in range(len(values))])

def clean_data(cohorts_dict, reference_dict):
    graph_dict = {}

    for dataset, variables in cohorts_dict.items():
        graph_dict[dataset] = {}
        for var, vals in variables.items():
            if var == 'Ethnicity':
                graph_dict[dataset][var] = multi_eth_change_format(vals,dataset)
            elif isinstance(list(vals.values())[0],dict):
                graph_dict[dataset][var] = nested_change_format(vals,var)
            else:
                graph_dict[dataset][var] = change_format(vals, var)

    ref_dict = {}
    for var,vals in reference_dict['2011 Census'].items():
        if var == 'Ethnicity':
            ref_dict[var] = eth_change_format(vals,'ethnicity')
        else:
            ref_dict[var] = change_format(vals, var)
            
    for var,vals in ref_dict.items():
        ref_dict[var]['percent'] = get_percents(list(vals['values']))

    for dataset, variables in graph_dict.items():
        for var,vals in variables.items():
            missing = {get_name(k,'missing'):[v[-1]]*(len(v) -1) for k,v in vals.items() if isinstance(v[0],int)}
            perc_dict = {get_name(k,'percent'): get_percents(v) for k,v in vals.items() if isinstance(v[0],int)}
            ref_pers = ref_dict[var]['percent']
            rel_dict = {get_name(k,'reletive'): get_reletives(v,ref_pers) for k,v in perc_dict.items()}
            rel_colours_dict = {}
            rep_or_not_legend_dict = {}
            rep_threshold_dict = {}
            for k,v in rel_dict.items():
                rep_threshold_dict[k + '_rep_threhsold'] = [100]*len(v)
                rel_colours = []
                rep_or_not = []
                for i in v:
                    if i < 95:
                        rel_colours.append('#FF112C')
                        rep_or_not.append('Under-represented')
                    elif i < 105:
                        rel_colours.append('#90C877')
                        rep_or_not.append('Well represented')
                    else:
                        rel_colours.append('#5FBFCE')
                        rep_or_not.append('Over-represented')
                rel_colours_dict[k + '_colours'] = rel_colours
                rep_or_not_legend_dict[k + '_representative_or_not'] = rep_or_not
            std_ref_dict = {get_name(k,'reference standardised'):
                                standardise_refs(v,ref_pers) for k,v in vals.items() if isinstance(v[0],int)}
            vals_short = {k:v[:-1] for k,v in vals.items()}
            desc_text = {'description text': ['this is description text for this variable'] * len(ref_pers)}
            ref_pers = {'ref percent': ref_pers}
            rel_text = {get_name(k,'text'):[reletivise_text(i) for i in v] for k,v in rel_dict.items()}
            if 'values' in vals_short.keys():
                cat_list = ['']
            else:
                cat_list = [re.sub('values','',i) for i in vals.keys() if 'values' in i]

            abs_rel_text = {get_name(i,
                                     'abs text'):absolute_reletivie_list(vals_short[str(i) + 'values'],
                                                                           std_ref_dict[str(i) + 'reference standardised']) for i in cat_list}
            abs_rel_text = {re.sub('  ',' ',k):v for k,v in abs_rel_text.items()}

            graph_dict[dataset][var]={**vals_short,
                                      **perc_dict,
                                      **rel_dict,
                                      **rel_colours_dict,
                                      **rep_or_not_legend_dict,
                                      **rep_threshold_dict,
                                      **std_ref_dict,**missing,**ref_pers,**desc_text,**rel_text,**abs_rel_text}

    return ref_dict, graph_dict

def spider_plot_source(spider_dict,addition):
    imp_keys = ['ref percent'] + [k for k in spider_dict.keys() if addition in k]
    test_spider = {re.sub(addition, '', k): v for k, v in spider_dict.items() if k in imp_keys}

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

    rel_values = np.array(test_spider['reletive'])
    values = np.array(test_spider['values'])

    ref_std = np.array(test_spider['reference standardised'])

    rep_lower_threshold_mapped = 95/(sum(rel_values) *2)
    rep_upper_threshold_mapped = 105/(sum(rel_values) *2)
    
    values = values / (sum(values) * 2)
    ref_std = ref_std / (sum(ref_std) * 2)
    rel_values = rel_values/(sum(rel_values) *2)

    rep_lower_threshold = np.array([rep_lower_threshold_mapped]*len(rel_values))
    rep_upper_threshold = np.array([rep_upper_threshold_mapped]*len(rel_values))

    x_val, y_val = radar_patch(values, theta)
    x_ref, y_ref = radar_patch(ref_std, theta)
    x_rel, y_rel = radar_patch(rel_values,theta)
    x_l_rep, y_l_rep = radar_patch(rep_lower_threshold,theta)
    x_u_rep, y_u_rep = radar_patch(rep_upper_threshold,theta)


    new_line_max = np.array([max(np.concatenate([values, ref_std]))] * len(values))
    new_x, new_y = radar_patch(new_line_max, theta)
    new_x_lines = [[0.5, i] for i in new_x]
    new_y_lines = [[0.5, i] for i in new_y]

    rel_new_line_max = np.array([max(rel_values)] * len(rel_values))
    rel_new_x, rel_new_y = radar_patch(rel_new_line_max, theta)
    rel_new_x_lines = [[0.5, i] for i in rel_new_x]
    rel_new_y_lines = [[0.5, i] for i in rel_new_y]

    source = {
        addition + 'x_vals':np.array(x_val),
        addition +'y_vals':np.array(y_val),
        addition +'x_ref':np.array(x_ref),
        addition +'y_ref':np.array(y_ref),
        addition +'x_rel':np.array(x_rel),
        addition +'y_rel':np.array(y_rel),
        addition +'x_l_rep':np.array(x_l_rep),
        addition +'y_l_rep':np.array(y_l_rep),
        addition +'x_u_rep':np.array(x_u_rep),
        addition +'y_u_rep':np.array(y_u_rep),
        addition +'x_lines':new_x_lines,
        addition +'y_lines':new_y_lines,
        addition +'rel_x_lines': rel_new_x_lines,
        addition +'rel_y_lines': rel_new_y_lines,
        addition +'labs_x_cords':np.array(new_x),
        addition +'labs_y_cords':np.array(new_y),
        addition +'rel_labs_x_cords':np.array(rel_new_x),
        addition +'rel_labs_y_cords': np.array(rel_new_y),
    }
    return source

def full_spider_source(data_dict):
    if 'values' in data_dict.keys():
        spider_dict = spider_plot_source(data_dict,'')
        source = {**data_dict,**spider_dict}
    else:
        var_list = [re.sub('values','',i) for i in data_dict.keys() if 'values' in i]
        source = data_dict.copy()
        for i in var_list:
            spider_dict = spider_plot_source(data_dict,i)
            source.update(spider_dict)
    return source

def update_graph_dict(g_dict):
    g_dict2 = g_dict.copy()
    for dataset in g_dict2.keys():
        for var in g_dict2[dataset].keys():
            if var == 'Ethnicity' or var == 'Socioeconomic Status':
                g_dict2[dataset][var] = full_spider_source(g_dict2[dataset][var])
    return g_dict2

def boxy_sanky(eth_dict,addition):
    perc = eth_dict[addition + 'percent']
    ref_p = eth_dict['ref percent']
    y_coords = [[80 if i == 0 else round(sum(perc[:i]),1),
                             round(sum(perc[:i + 1]),1),
                             round(sum(ref_p[:i + 1]),1),
                             80 if i == 0 else round(sum(ref_p[:i]),1)] for i in range(len(perc))]

    return y_coords

def ethnicity_tips(eth_dict):
    out_str = []
    for eth in eth_dict.values():
        eth_str = str()
        for k,v in eth.items():
            if v > 0:
                tip =k + ':' + str(v) + ' '
                eth_str = eth_str + tip
        out_str.append(eth_str)
    out_str = out_str[:-1]
    del out_str[3]
    return out_str


if __name__ == '__main__':

    with open('data/raw/cohort_demographics_test_data.json', 'r') as fb:
        cohorts_dic = json.load(fb)

    with open('data/raw/Reference_population.json', 'r') as fb:
        reference_dict = json.load(fb)

    ref_dict, graph_dict = clean_data(cohorts_dic, reference_dict)
    graph_dict2 = update_graph_dict(graph_dict)
    graph_dict2['UK Biobank']['Text'] = 'The UK Biobank is a prospective cohort study that recruited adults aged between 40-69 years in the UK in 2006-2010. People were invited to participate by mailed invitations to the general public living within 25 miles of one of the 22 assessment centres in England, Scotland and Wales (there was a response rate of 5.5%). '
    graph_dict2['ALSPAC']['Text'] = 'The Avon Longitudinal Study of Children and Parents (ALSPAC) is a prospective cohort study which recruited pregnant women living in the South West of England during 1990-1992. It aims to understand how genetic and environmental factors influence health and development in parents and children by collecting information on demographics, lifestyle behaviours, physical and mental health. The parents and children have been followed up since recruitment through questionnaires, and a subset completed additional assessments (e.g. ‘Focus on Mothers’) which collected anthropometric measurements and biological samples.'
    graph_dict2['ALSPAC']['Age']['description text'] =['At recruitment, the mother was asked to describe her age and that of her partner. The children were obviously all born shortly after their mothers were invited to join the study, so their age at recruitment is 0 years. Overtime, subsequent data was collected at different time points, providing a longitudinal perspective on key health and lifestyle characteristics. So, whilst these labels reflect the baseline characteristics, it does not capture any changes during the participants’ life course (for example when the children are grown-up, their socioeconomic status may be different).'] * len(graph_dict2['ALSPAC']['Age']['Age'])
    graph_dict2['ALSPAC']['Ethnicity']['description text'] = ['The mother was asked to describe the ethnic origin of herself, her partner and her parents in a questionnaire. There were 9 possible ethnicity categories: white, Black/Caribbean, Black/African, Black/other, Indian, Pakistani, Bangladeshi, Chinese, Other. Most research using this data derived the childs ethnic background as ‘white’ (if both parents were described as white) or ‘non-white’ (if either parent was described as any ethnicity other than white). The 9 categories for ethnicity offer a greater level of granularity than many other cohort studies. However, there are far more ethnic groups represented in the UK, and often people do not identify with one ethnicity. These groups also get aggregated into just 2 categories (white or non-white) for the child’s ethnicity, meaning that it may be difficult to understand any nuances or differences in health and well-being related to ethnic background.  Often larger but fewer categories are used for analysis to ensure the sample size is large enough for statistical signifacince.'] * len(graph_dict2['ALSPAC']['Ethnicity']['Ethnicity'])
    for dataset in graph_dict2.keys():
        graph_dict2[dataset]['Ethnicity']['description text'] = ['The 5 ethnicities are the groups which all datasets have in common. Some datasets did collect more granular data (up to 16 categories) but to compare the representativeness between datasets, the data has been grouped to these higher-level categories.']* len(graph_dict2[dataset]['Ethnicity']['Ethnicity'])
        graph_dict2[dataset]['Socioeconomic Status']['description text'] =['Social class based on Occupation (formerly the UK Registrar General’s occupational coding) has been used across datasets as an indicator of socioeconomic status. The categories are<br><ul><li>V (unskilled)</li><li>IV (semi-skilled manual)</li><li>III (skilled manual)</li><li>III (non-manual)</li><li>II (managerial and technical)</li><li>I (professional)</li></ul>'] * len(graph_dict2[dataset]['Socioeconomic Status']['Socioeconomic Status'])
        if 'values' in graph_dict2[dataset]['Ethnicity'].keys():
            var_list = ['']
        else:
            var_list = [re.sub('values', '', i) for i in graph_dict2[dataset]['Ethnicity'].keys() if 'values' in i]
        boxy_y = {i + 'y_coords': boxy_sanky(graph_dict2[dataset]['Ethnicity'],i) for i in var_list}
        graph_dict2[dataset]['Ethnicity'].update(boxy_y)
        graph_dict2[dataset]['Ethnicity']['x_coords'] = [[0,0,100,100] for i in range(len(graph_dict2[dataset]['Ethnicity']['Ethnicity']))]
        graph_dict2[dataset]['Ethnicity']['colours'] = ["#003667","#ed6b00","#87f5fb","#a882dd","#721817"]

    for dataset,variables in cohorts_dic.items():
        print(dataset)
        if dataset in ['UK Biobank','National Child Development Study', 'Whitehall II study', 'HES']:
            graph_dict2[dataset]['Ethnicity']['tips'] = ethnicity_tips(variables['Ethnicity'])
        else:
            tips = {str(k) + ' tips':ethnicity_tips(v) for k,v in variables['Ethnicity'].items()}
            graph_dict2[dataset]['Ethnicity'].update(tips)


