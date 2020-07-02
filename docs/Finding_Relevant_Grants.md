# Finding Relevant Grants

The first step in the Nutrition Labels project is to classify Wellcome grants as producing datasets/models/tools. Since there are 17,000 grants in the publically avaiable grants data from 2005 to 2019 this would be very time consuming to do by manually reading all the grants descriptions. Furthermore the grants description may not be enough to tell you whether the grant produced such an outcome.

Thus, we will create a model to predict whether a grant was likely to contribute to a dataset or tool ('relevant') or not. The initial step to this is to create a training data set of grants we know have and haven't produced relevant outcomes.

## Relevant grants

*to do: write a bit about the current state of what constitutes a 'relevant' grant (tool, model, dataset), list assumptions*

## Raw data

There were three possible data sources we looked at when tagging whether a grant was relevant or not.

1. Grant descriptions
2. ResearchFish questionnaire data
3. EPMC publication abstracts from publications with Wellcome funding acknowledged

### 1. Grant descriptions

The first step was reading the grant descriptions and trying to discern if they were relevant or not. It became clear that a really small proportion were relevant. The grants were filtered first to speed up this process, the filters applied were:

*To do: explain the filtering*

873 of these grants were tagged as
- 1 = Definitely relevant
- 2 = Possibly relevent
- 3 = not relevant but initially looked relevant 
- 4 = really not relevant

This tagged data is in `data/raw/wellcome-grants-awarded-2005-2019_manual_edit.csv`.

### 2. ResearchFish questionnaire data

Using the ResearchFish platform we've asked our grantholders over the last couple of years to self-report outcomes of their grants. There are a few fields for this which might flag which grants have outputted datasets/tools, these are in fortytwo (`FortyTwo_Denormalised.[ResearchFish]`) and the table names are:
- Medical Products, Interventions & Clinical Trials (fortytwo table: `MedicalProductsAndInterventions`)
- Research Databases & Models (fortytwo table: `ResearchDatabaseModels`)
- Research Tools & Methods (fortytwo table: `ResearchToolsAndMethods`)
- Software & Technical Products (fortytwo table: `SoftwareAndTechnicalProducts`)
- Other Outputs & Knowledge/Future Steps (fortytwo table: `OtherOutputsAndKnowledge`)

This yielded 1264 outcome reports, all these outcome descriptions were read and tagged as being
- 1 = Relevant tool
- 2 = Relevant model
- 3 = Relevant data set
- 4 = More information needed
- 5 = Not relevant

This tagged data is in `data/raw/ResearchFish/research_fish_manual_edit.csv`.

### 3. EPMC publication abstracts

A final source to find relevant grants was to look for certain keywords in Wellcome acknowledged EPMC publication abstracts. This data was again from fortytwo (`FortyTwo_Denormalised.[EPMC].[PublicationFull]`).

The list of keywords used to filter this data is:
["platform", "software", "web resource", "pipeline", "toolbox", "data linkage", "database linkage", "record linkage", "linkage algorithm", "python package", "r module", "python script", "web tool", "web based tool"]

This yielded 3129 publications. 619 of these were tagged as being

- 1 = Relevant tool
- 2 = Relevant model
- 3 = Relevant data set
- 4 = More information needed
- 5 = Not relevant
- 6 = Edge case

This tagged data is in `data/raw/EPMC_relevant_tool_pubs_manual_edit.csv`.

## Compiling the training data

By running `python nutrition_labels/grant_data_processing.py` the three sets of tagged training data are combined with the publically available grants data `data/raw/wellcome-grants-awarded-2005-2019.csv`. This is outputted in `data/processed/training_data.csv`.

Not all the tagged data from ResearchFish (5 grants) or the EPMC publications (7 grants) were able to be linked back to the grants data since the grant reference wasn't found. Also for the EPMC data often only the 6 digit grant reference was given, in these cases we decided to combine this with the most recent grant in the grants data.

This dataset comprises of a grant reference, the grant title and description and also the tagged code.

### Training data summary - 30th June 2020 

| Tag code | Meaning | Number of grants |
|---|---|--- | 
| 1 | Relevant tool | 41 |
| 2 | Relevant model | 14 |
| 3 | Relevant dataset | 13 |
| 4 | Not relevant | 791 |


## `grant_tagger.py`


In `grant_tagger.py` we train a model to predict whether a grant is relevant or not. For this we collapsed the classification into binary, where 
- 1 = class was relevant tool (1), relevant model (2) or relevant dataset (3)
- 0 = class was not relevant (4)



