import re
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix

def clean_grants_data(
    old_grant_data, text_column="Description", grant_id_column="Internal ID"
):
    """
    Clean grant descriptions of html and remove any duplicates
    """
    dont_include_text = ["Not available", ""]
    grant_data = old_grant_data.copy()
    grant_data.dropna(subset=[text_column], inplace=True)
    grant_data[text_column] = grant_data[text_column].apply(remove_useless_string)
    grant_data = grant_data[~grant_data[text_column].isin(dont_include_text)]
    grant_data.drop_duplicates(grant_id_column, inplace=True)
    grant_data["Internal ID 6 digit"] = grant_data[grant_id_column].apply(
        lambda x: re.sub("/.*", "", x)
    )

    # After dropping rows you need to reset the index
    grant_data.reset_index(inplace=True)
    return grant_data

def remove_useless_string(string):
    '''
    cleans the grant descriptions of artifacts such as <br />
    :param string: description string
    :return: clean string
    '''

    soup = BeautifulSoup(string, features="lxml")
    string_out = soup.get_text()
    string_out = string_out.replace('\n', ' ')
    string_out = string_out.replace('\xa0', ' ')
    return(string_out)

def only_text(string):
    '''
    removes non-alphanumeric characters and spaces to increase matching
    :param string: description string
    :return: clean string
    '''
    string = re.sub(' |\W','',string)
    return(string)

def pretty_confusion_matrix(y, y_predict, labels=[0,1], text=['actual', 'predicted']):
    '''
    sklearn's confusion matrix doesn't give informative row and col labels
    Confusion matrix whose i-th row and j-th column entry indicates
    the number of samples with true label being i-th class and prediced label being j-th class.
    '''

    cm = pd.DataFrame(confusion_matrix(y, y_predict, labels = labels))
    if text:
        cm.rename(
            index={i:f'{text[0]} tag {label}' for i, label in enumerate(labels)},
            columns={i:f'{text[1]} tag {label}' for i, label in enumerate(labels)},
            inplace=True)
    else:
        cm.rename(
            index={0:'Actually not relevant', 1:'Actually relevant'},
            columns={0:'Predicted not relevant', 1:'Predicted relevant'},
            inplace=True)
    return cm
