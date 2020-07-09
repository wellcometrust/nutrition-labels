import re
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

