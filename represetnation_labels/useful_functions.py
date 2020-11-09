import json
import re

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
    if name_key == 'values' or name_key == 'percent':
        return addition
    else:
        alter_name = re.sub(' values','',name_key)
        alter_name = re.sub(' percent','', alter_name)
        return str(alter_name) + ' ' + (addition)

def reletivise_text(perc):
    string =  'there is ' + str(round(perc)) + '% of this group needed to be representative'
    return(string)

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
            std_ref_dict = {get_name(k,'reference standardised'):
                                standardise_refs(v,ref_pers) for k,v in vals.items() if isinstance(v[0],int)}
            vals_short = {k:v[:-1] for k,v in vals.items()}
            desc_text = {'description text': ['this is description text for this variable'] * len(ref_pers)}
            ref_pers = {'ref percent': ref_pers}
            rel_text = {get_name(k,'text'):[reletivise_text(i) for i in v] for k,v in rel_dict.items()}
            graph_dict[dataset][var]={**vals_short,
                                      **perc_dict,
                                      **rel_dict,**std_ref_dict,**missing,**ref_pers,**desc_text,**rel_text}

    return ref_dict, graph_dict

if __name__ == '__main__':

    with open('data/raw/cohort_demographics_test_data.json', 'r') as fb:
        cohorts_dic = json.load(fb)

    with open('data/raw/Reference_population.json', 'r') as fb:
        reference_dict = json.load(fb)

    ref_dict, graph_dict = clean_data(cohorts_dic, reference_dict)

