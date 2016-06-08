from bs4 import BeautifulSoup
import requests
import json
import re

def extractLabels():
    """Extracts the numeric values and the respective description of the labels

    The description and numbers of the labels used for land cover 
    classification in the NLCD 2011 are found on the website:
    http://www.mrlc.gov/nlcd11_leg.php

    Returns:
        dict: Numbers as keys, (class, short description, long description)-tuple as value

    """
    theDict = {}
    r = requests.get('http://www.mrlc.gov/nlcd11_leg.php')
    soup = BeautifulSoup(r.text, 'html.parser')
    # table containing the labels and their descriptions
    table = soup.find_all('table')[2]

    # used to  keep track of current class (e.g. Water, Developed)
    currentClass = ''
    for row in table.find_all('tr'):
        # get rid of non-utf chars
        string = ''.join(i for i in str(row) if ord(i) < 128)
        # look for class
        matchClass = re.search(r'<strong>(\w+)</strong></td>', string)
        if matchClass:
            currentClass = matchClass.group(1)
        # look for number and description
        match = re.search(r'>(\d\d)</td>.+<strong>(.+)</strong>\s?-?\s?([^\r\n]+)\r?\n?</td>', string)
        if match:
            number = int(match.group(1))
            shortDescription = match.group(2)
            longDescription = match.group(3).strip('\u0096')
            # store value in dict
            theDict[number] = (currentClass, shortDescription, longDescription)
        else:
            print str(row)
    return theDict



def writeDictToJson(myDict, path):
    """Save the dictionary as JSON to specified output file

    Yes, I wrote a function for this, it is 3am and I did not fucking 
    realize until afterwards how easy it is to do this with json.

    Args:
        myDict (dict): dictionary to be saved
        path (str): path to output file
    """
    with open(path, 'w') as outfile:
        json.dump(myDict, outfile, sort_keys=True, indent=4)


if __name__ == '__main__':
    labelDict = extractLabels()
    writeDictToJson(labelDict, 'NLCD_labels.json')
