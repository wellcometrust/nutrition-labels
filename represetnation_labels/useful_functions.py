import pandas as pd
import json



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

def multi_eth_change_format(eth_dict):
    if list(eth_dict.keys()) == ['White', 'Black', 'Asian', 'Chinese', 'Mixed', 'Other', 'Missing']:
        return eth_change_format(eth_dict,'Ethnicity')
    else:
        dict_copy = {k: eth_change_format(v, k) for k, v in eth_dict.items()}
        dict_copy = {k: v['values'] for k, v in dict_copy.items()}
        dict_copy['Ethnicity'] = ['White', 'Black', 'Asian', 'Mixed', 'Other', 'Missing']
        return dict_copy

def multi_change_format(multi_dict,name):
    fst_key = list(multi_dict.keys())[0]
    variable = list(multi_dict[fst_key].keys())
    new_dict = {k:[values for values in v.values()] for k,v in multi_dict.items()}
    new_dict[name] = variable
    return new_dict

if __name__ == '__main__':

    with open('data/raw/cohort_demographics_test_data.json', 'r') as fb:
        cohorts_dic = json.load(fb)

    with open('data/raw/Reference_population.json', 'r') as fb:
        reference_dict = json.load(fb)

    graph_dict = {}

    for dataset, variables in cohorts_dic.items():
        graph_dict[dataset] = {}
        for var, vals in cohorts_dic[dataset].items():
            if var == 'Ethnicity':
                graph_dict[dataset][var] = multi_eth_change_format(vals)
            elif isinstance(list(vals.values())[0],dict):
                graph_dict[dataset][var] = multi_change_format(vals,var)
            else:
                graph_dict[dataset][var] = change_format(vals, var)

    ref_dict = {}
    for var,vals in reference_dict['2011 Census'].items():
        if var == 'Ethnicity':
            ref_dict[var] = eth_change_format(vals,'ethnicity')
        else:
            ref_dict[var] = change_format(vals, var)

