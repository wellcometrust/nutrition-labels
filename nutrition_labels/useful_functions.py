import re
import x
from bs4 import BeautifulSoup

def remove_useless_string(string):
    '''
    cleans the grant descriptions of artifacts such as <br />
    :param string: description string
    :return: clean string
    '''

    soup = BeautifulSoup(string)
    string_out = soup.get_text()
    string_out = string_out.strip('\n')
    string_out = string_out.strip('\xa0')
    string_out = re.sub('  ','',string_out)
    return(string_out)

def only_text(string):
    '''
    removes non-alphanumeric characters and spaces to increase matching
    :param string: description string
    :return: clean string
    '''
    string = re.sub(' |\W','',string)
    return(string)

def pretty_confusion_matrix(y, y_predict, labels=[0,1]):
    '''
    sklearn's confusion matrix doesn't give informative row and col labels
    Confusion matrix whose i-th row and j-th column entry indicates
    the number of samples with true label being i-th class and prediced label being j-th class.
    '''

    cm = pd.DataFrame(confusion_matrix(y, y_predict, labels = labels))

    cm.rename(
        index={0:'Actually not relevant', 1:'Actually relevant'},
        columns={0:'Predicted not relevant', 1:'Predicted relevant'},
        inplace=True)
    return cm
