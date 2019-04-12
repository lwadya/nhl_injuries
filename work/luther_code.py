import os
import re
import itertools
import pickle
import pandas as pd
from bs4 import BeautifulSoup


def read_profile_links(driver):
    '''Scrapes all player profile links from TSN.ca's main player page

    Args:
        driver (selenium webdriver): a valid webdriver

    Returns:
        list: player profile URLs
    '''
    main_url = 'https://www.tsn.ca/nhl/players'
    link_pattern = '/nhl/player-bio/[^"]+'
    disabled_class = 'ng-scope disabled'

    # Iterates through all pages of player profile links
    driver.get(main_url)
    links = []
    while True:
        links.extend(re.findall(link_pattern, driver.page_source))
        next_button = driver.find_element_by_css_selector('a.next.ng-scope')
        if next_button.find_element_by_xpath('..').get_attribute('class') == disabled_class:
            break
        next_button.click()

    return links

def read_player_profile(driver, player_url):
    '''Scrapes a player's injury, transaction, and suspension history from TSN.ca

    Args:
        driver (selenium webdriver): a valid webdriver
        player_url (str): url that contains player profile data

    Returns:
        list: player event data as a list of lists, each nested list containing
              [player name, birth date, date, event description]
    '''
    driver.get(player_url)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    roster_updates = []

    # Player Name is stored in list items with specific classes
    first_name = soup.find('li', {'class':'first-name ng-binding'}).text
    last_name = soup.find('li', {'class':'last-name ng-binding'}).text
    player_name = f'{first_name} {last_name}'

    # Birth Date is stored in a span with a specific class
    birth_date = soup.find('span', {'class':'value-desc ng-binding'}).text

    # Date and Event are stored in two spans under 'tr' tags with specific 'ng-repeat' values
    rows = soup.find_all('tr', {'ng-repeat':'rosterMoves in PlayerBio.RosterMoves'})
    for row in rows:
        spans = row.find_all('span')
        update = [span.text for span in spans]
        if update:
            roster_updates.append([player_name, birth_date] + update)

    return roster_updates

def profiles_to_dfs(player_profiles):
    '''Converts a list of player profiles into a DataFrame containing just injury data

    Args:
        player_urls (str list): nested lists of individual player profiles

    Returns:
        DataFrame: player name data with columns [Name, Birth_Date]
        DataFrame: player injury data with columns
                   [Name, Birth_Date, Injury_Date, Report, Games_Missed, Cause]
    '''
    columns = ['Name', 'Birth_Date', 'Injury_Date', 'Report']
    df = pd.DataFrame(list(itertools.chain(*player_profiles)), columns=columns)
    df['Birth_Date'] = pd.to_datetime(df['Birth_Date'], format='%Y/%m/%d')
    names_df = df[['Name', 'Birth_Date']].drop_duplicates().reset_index(drop=True)

    # Isolate reports of missed regular season games only
    df['Report'] = df['Report'].str.lower()
    df = df[df['Report'].str.contains('(?<=missed)\D*\d\d*')]
    mask = ((~df['Report'].str.contains('playoff') &
             ~df['Report'].str.contains('round')) |
             df['Report'].str.contains('regular'))
    df = df[mask]

    df['Injury_Date'] = pd.to_datetime(df['Injury_Date'], format='%b %d, %Y')
    # Assume the number of missed games is the first numerical data in reports
    df['Games_Missed'] = df['Report'].str.extract('(\d\d*)').astype(int)
    # Cause of missed games is everything between parentheses
    df['Cause'] = df['Report'].str.findall('(?<=\().*(?=\))').str.join('')

    # Remove any entries that are not injuries
    df['Cause'] = df['Cause'].str.replace('\).*', '')
    non_injuries = [
        'appendectomy',
        'appendicitis',
        'bereavement list',
        'bronchitis',
        'conditioning',
        'colon surgery',
        'cyst surgery',
        'disciplinary',
        'flu',
        'food poisoning',
        'gastrointestinal virus',
        'h1n1 virus',
        'heart',
        'heart ailment',
        'heart arrhythmia',
        'heart surgery',
        'hepatitis a',
        'illness',
        'infected cut on leg',
        'infected elbow',
        'infection',
        'inflamed tonsils',
        'intra-abdominal lymphoma (non-hodgkin)',
        'irregular heartbeat',
        'kidney stones',
        'leukemia',
        'mononucleosis',
        'multiple sclerosis treatment',
        'mumps',
        'nasal surgery',
        'nervous system disorder',
        'nhl-nhlpa susbstance abuse program',
        'non-roster player',
        'not with team',
        'personal reasons',
        'rash',
        'respiratory',
        'rest',
        'root canal',
        'sick',
        'sinus',
        'sinus infection',
        'sinusitis',
        'skin infection',
        'stomach ailment',
        'stomach flu',
        'stomach virus',
        'strep throat',
        'suspended by nhl',
        'suspended by team',
        'suspension',
        'suspension/flu',
        'thyroid',
        'viral illness',
        'viral infection',
        'virus',
        'visa issues'
    ]
    df = df[~df['Cause'].isin(non_injuries)]
    return names_df, df.reset_index(drop=True)

def var_to_pickle(var, filename):
    '''Writes the given variable to a pickle file

    Args:
        var (any): variable to be written to pickle file
        filename (str): path and filename of pickle file

    Returns:
        None
    '''
    try:
        with open(filename, 'wb') as f:
            pickle.dump(var, f)
    except:
        print(f'Failed to save pickle to \'{filename}\'')
    return

def read_pickle(filename):
    '''Reads the given pickle file

    Args:
        filename (str): path and filename of pickle file

    Returns:
        any: contents of pickle file if it exists, None if not
    '''
    output = None
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                output = pickle.load(f)
        except:
            print(f'Failed to load pickle from \'{filename}\'')
    return output
