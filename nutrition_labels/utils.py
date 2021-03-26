import re
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix

# Common synopsis/description fields which mean 'No data given'
not_available_aliases = {
        'Not available': None,
        'Summary not available': None,
        'NA': None,
        'N/A': None,
        'no abstract available.': None,
        'To be submitted later': None,
        'A': None,
        '': None,
    }


def clean_grants_data(
    old_grant_data, text_column="Description", grant_id_column="Internal ID"
):
    """
    Clean grant descriptions of html and remove any duplicates
    """
    grant_data = old_grant_data.copy()
    grant_data[text_column] = grant_data[text_column].apply(clean_string)
    grant_data.dropna(subset=[text_column], inplace=True)
    grant_data.drop_duplicates(grant_id_column, inplace=True)
    grant_data["Internal ID 6 digit"] = grant_data[grant_id_column].apply(
        lambda x: re.sub("/.*", "", x)
    )

    # After dropping rows you need to reset the index
    grant_data.reset_index(inplace=True)
    return grant_data

def clean_string(string):
    '''
    Cleans the grant descriptions of artifacts such as <br />
    and converts all 'No data given' synonyms into None
    :param string: description string
    :return: clean string
    '''

    if string:
        soup = BeautifulSoup(string, features="lxml")
        string_out = soup.get_text()
        string_out = string_out.replace('\n', ' ')
        string_out = string_out.replace('\xa0', ' ')

        if string_out in not_available_aliases:
            string_out = not_available_aliases[string_out]
    else:
        # If None is inputted output None
        string_out = string

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
