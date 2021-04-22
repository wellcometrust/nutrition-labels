# Top UK health datasets

For the data representation labels we need to first find the top UK health datasets. This is done by looking for commonly mentioned datasets in publications associated with the tech grants.

## Getting the references of the tech grants' publications

Running

```
python nutrition_labels/finding_tech_grant_refs.py
```
will find all the PMIDs of publications which acknowledge any grant numbers in the tech grant we identified from the ensemble model (given in 'data/processed/ensemble/200907/ensemble_results.csv'). We found that 1138 of the 1790 tech grants (64%) were acknowledged in publications.

Then it uses the EPMC client to get all the references from each of these publications. This creates two useful outputs - `data/processed/top_health_datasets/tech_grants_references_dict_2020-09-16.json` information about each of the references found, and `data/processed/top_health_datasets/tech_grants_reference_ids_counter_2020-09-16.json` a counter of how many times each reference was found.

## Most common health datasets

This data is then used to find the most common health data sets by first running:

```
python nutrition_labels/finding_most_common_references.py
```

This uses data extracted from EPMC (`data/processed/top_health_datasets/tech_grants_references_dict_2020-09-16.json` and `data/processed/top_health_datasets/tech_grants_reference_ids_counter_2020-09-16.json`) which has all the references from all the papers produced from tech grants.
The list of most referenced papers is then filtered to find ones only relating to tools using terms: 'cohort','profile','resource','biobank','Cohort','Profile','Biobank','Resource'. This is outputted in `data/processed/health_data_sets_list.csv`.
This is then manually filtered to find only UK health data sets which can be found in `data/processed/top_health_datasets/health_data_sets_list_manual_edit.csv`.

To find the 5 most cited health data sets in total and last five years run: 

```
python nutrition_labels/finding_most_common_cohort.py
```

This uses the EPMC client to find the number of citations each paper in the `health_data_sets_list_manual_edit.csv` dataset above has and returns only the ones since 2015. 

## Results

The results are: 

| Study | Total citations| Recent citations |
|---|---|--- |
| UK biobank: an open access resource for identifying the causes of a wide range of complex diseases of middle and old age. | 814 | 814 |
| Cohort Profile: the 'children of the 90s'--the index offspring of the Avon Longitudinal Study of Parents and Children | 903 | 734 |
| Data Resource Profile: Clinical Practice Research Datalink (CPRD). | 664 | 664 |
| Cohort Profile: the Avon Longitudinal Study of Parents and Children: ALSPAC mothers cohort. | 609 | 518 |
| Cohort profile: 1958 British birth cohort (National Child Development Study). | 418 | 176 |
| Cohort Profile: the Whitehall II study. | 378 | 146 |
