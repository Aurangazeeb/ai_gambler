import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from collections import defaultdict
import pandas as pd
import numpy as np
import string
from itertools import combinations, permutations
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier
from xgboost import XGBRFClassifier
from sklearn.model_selection import train_test_split
import os


class PlayShop():
    
    driver_path = 'E:\\Essential Softwares\\chrome_driver\\chromedriver.exe'
#     webdriver = webdriver
    def __init__(self, website, headless = True, incognito = True, *browser_args, exec_path = driver_path):
        self.name = website
        self._webdriver = webdriver
        self._exec_path = exec_path
        self._driver_options = Options()
        if headless:
            self._driver_options.add_argument('--headless')
        if incognito:
            self._driver_options.add_argument('--incognito')
        if browser_args is not None:
            for arg in browser_args:
                self._driver_options.add_argument(arg)
        self.driver = self._webdriver.Chrome(executable_path = self._exec_path, options = self._driver_options)
        
    def make_page_constants(self, login_xpaths, \
                            table_xpaths, \
                            page_base_xpath, \
                            time_base_xpath,\
                            num2colormap, \
                            server_base, \
                            server_names = ['parity','sapre','bcone','emerd'], \
                            color_list = ['green','red','violet'], \
                            num_entries = 10, \
                            n_numbers = 10
                           ):
        self.next_clicks = 0
        self.login_dict = login_xpaths
        self.table_dict = table_xpaths
        self.page_dict = page_base_xpath
        self.time_dict = time_base_xpath
        self.color_list = color_list
        self.total_colors = self.color_list
        self.num_entries = num_entries
        self.nummap = num2colormap
        self.current_server = 1
        self.servers = server_names
        self.server_dict = server_base
        
        #color combo available depends on the nummap
        total_colorarray = np.array(list(set(self.nummap.values()))).reshape(-1,1)
        self.n_colorcombo = total_colorarray.size
        self.n_numbers = n_numbers
        self.color_encoder = OrdinalEncoder(dtype = int).fit(total_colorarray)
        
    def _goto_shop(self, sleeptime= 3):
        print(f'Fetching {self.name} ...')
        self.driver.get('https://' + self.name)
        time.sleep(sleeptime)

        try:
            wait = WebDriverWait(self.driver, 5)
            #the send TimeOUt exception if not found
            wait.until(expected_conditions \
                       .presence_of_element_located((By.XPATH,\
                                                     self.login_dict['notice_button_xpath'])))
            close_button = self.driver.find_element_by_xpath(self.login_dict['notice_button_xpath'])
            close_button.click()
            time.sleep(sleeptime - 1)

        except:
            pass
        
        login_tab = self.driver.find_element_by_xpath(self.login_dict['login_tab_xpath'])
        login_tab.click()
        time.sleep(sleeptime)
        print('Fetch completed...')
        
    
    def _login(self, credentials, sleeptime = 3):
        '''
        login xpaths : xpath username, password, login button resp.
        credentials : username, password tuple
        '''

        print('Logging in ...')
        mob_number_field = self.driver.find_element(By.XPATH, self.login_dict['username_xpath'])
        password_field = self.driver.find_element_by_xpath(self.login_dict['password_xpath'])
        login_button = self.driver.find_element(By.XPATH, self.login_dict['login_button_xpath'])
        mob_number_field.send_keys(credentials[0])
        time.sleep(sleeptime)
        password_field.send_keys(credentials[1])
        time.sleep(sleeptime)
        login_button.click()
        time.sleep(sleeptime)
        print('You are logged in !')
    
    
    def _scoreboard(self, sleeptime = 2):
        print('Navigating to scoreboard...')
        wait_obj = WebDriverWait(self.driver, 10)
        wait_obj.until(expected_conditions. \
                       presence_of_element_located \
                       ((By.XPATH, self.login_dict['score_tab_xpath'])))
        win_tab = self.driver.find_element_by_xpath(self.login_dict['score_tab_xpath'])
        win_tab.click()
        time.sleep(sleeptime)
        print(f'Arrived at scoreboard of {self.servers[self.current_server - 1]} server !')
        
    def select_server(self, server_num = 1, num_servers = 4):
        if server_num in range(1, num_servers + 1):
            server_name = self.servers[server_num - 1]
            if server_num != self.current_server:
                self.current_server = server_num
                server_xpath = self.server_dict['server_base'].format(server_num)
                self.driver.find_element(By.XPATH, server_xpath).click()
                print(f'Switched to server : {server_name}')
            else:
                print(f'You are on {server_name} !')
        else:
            server_numbers = list(range(1, len(self.servers) + 1))
            raise Exception(f'No such server. Try {server_numbers}')

        '''
        Uses all three internal APIs above
        '''
    def visit(self, credentials, sleeptime = 2.5):
        self._goto_shop(sleeptime)
        self._login(credentials, sleeptime)
        self._scoreboard(sleeptime)
        self._make_next_button()
    
#     shopname = self.name
        
    
    def _make_next_button(self, elem_num = 2):
        self.next_bttn_xpath = self.page_dict['page_base'].format(elem_num)
        self.next_button = self.driver.find_element(By.XPATH, self.next_bttn_xpath)
        
    def _goto_next_page(self, sleeptime = 0.5):
        self.next_button.send_keys(Keys.ENTER)
        self.next_clicks += 1
        current_num = int(self.next_button.text)
        if current_num < 4:
            self._make_next_button(current_num + 1)
        if current_num >= 4:
            self._make_next_button(3)
        time.sleep(sleeptime)

        #############################################
        
    def reload(self, wait = 1):
        self.driver.refresh()
        time.sleep(wait)
        self._make_next_button()
        
    def _get_timestamp(self, row_number, elem_index = 1, driver_wait= 3):
        timestamp_dict = defaultdict(list)
        timestamp_xpath = self.table_dict['timestamp'].format(elem_index, row_number)
        wait = WebDriverWait(self.driver, driver_wait)
        wait.until(expected_conditions.presence_of_element_located((By.XPATH, timestamp_xpath)))
        table_entry = self.driver.find_element_by_xpath(timestamp_xpath)
        timestamp_dict['timestamp'].append(table_entry.text)
        return timestamp_dict
    
    def _get_price(self, row_number, elem_index = 2):
        entry_dict = defaultdict(list)
        price_xpath = self.table_dict['price'].format(elem_index, row_number)
        price_elem = self.driver.find_element(By.XPATH, price_xpath)
        entry_dict['price'].append(int(price_elem.text))
        return entry_dict
    
    def _get_number(self, row_number, elem_index = 3):
        entry_dict = defaultdict(list)
        xpath = self.table_dict['number'].format(elem_index, row_number)
        elem = self.driver.find_element(By.XPATH, xpath)
        entry_dict['number'].append(int(elem.text))
        return entry_dict
    
    
    
    def _extract_a_row(self, row_index):
        
        '''
        Returns : dictionary of one row of table

        '''
        timestamp_dict = self._get_timestamp(row_index)
        try:
            price_dict = self._get_price(row_index)
        except:
            price_dict = defaultdict(list)
            price_dict['price'].append(0)
            one_row = dict(timestamp_dict, **price_dict)
            number_dict = defaultdict(list)
            number_dict['number'].append(0)
            one_row = dict(one_row, **number_dict)
            return one_row
        
        one_row = dict(timestamp_dict, **price_dict)
        number_dict = self._get_number(row_index)
        one_row = dict(one_row, **number_dict)
        return one_row

    def _merge_rows(self, current_dict, new_row_dict):
        key_list = list(current_dict.keys())
        for key in key_list:
            current_dict[key].append(new_row_dict[key][0])
        return current_dict

    def _extract_page(self, from_row = 1, num_entries = 10):
        row_dict = self._extract_a_row(from_row)
    #     print(pd.DataFrame(row_dict))
        for i in range(from_row + 1, num_entries + 1):
            new_row = self._extract_a_row(i)
            row_dict = self._merge_rows(row_dict, new_row)
        return row_dict
    
    def _merge_pages(self, current_page_dict, new_page_dict):
        key_list = list(current_page_dict.keys())
        for key in key_list:
            current_page_dict[key].extend(new_page_dict[key])
        return current_page_dict
    
    def _pull_pages(self, page_count, sleeptime = 1):
    
        '''
        @sleeptime - waits this much second after each page extraction
        '''

        #extract current page
        current_dict = self._extract_page()
        time.sleep(sleeptime)
        if(page_count > 1):
            for _ in range(2, page_count+1):
                #go to next page
                self._goto_next_page(sleeptime * 0.7)
                #extract next page
                new_page_dict = self._extract_page()
                #merge with existing page
                current_dict = self._merge_pages(current_dict, new_page_dict)
                time.sleep(sleeptime)
        return current_dict
    
    def savedata(self, df):
        if(self._quality(df) == 'nice'):
            day, month, date, clock, year = time.ctime().split()
            clock = '_'.join(clock.split(':'))
            time_format = '_'.join([clock, day, date, month, year])
            current_server_name = self.servers[self.current_server - 1]
            df.to_csv(self.name + '_' + current_server_name + '_' + time_format +'.csv', index = False)
            print(f'{self.name} dataset saved ...!')
        else:
            print('Aborting save due to bad data quality !')
        
        
    def _time_left(self, at_shop = False, driver_wait = 10, impl_wait = 2):
        try:
            wait = WebDriverWait(self.driver, driver_wait)
            wait.until(expected_conditions.presence_of_element_located((By.XPATH, self.time_dict['time_base'])))
        except:
            self.driver.implicitly_wait(impl_wait)
        time_string = self.driver.find_element_by_xpath(self.time_dict['time_base']).text
        time_remaining = self._str2time(time_string)
        if at_shop:
            time_remaining = time_string
        return time_remaining

    def _str2time(self, time_string):
#         print('time string :', time_string)
        try:
            split_time = time_string.split(':')
        except:
            split_time = time_string.split('.')
            
#         print('time split :', split_time)
        minutes, seconds = [int(float(num)) for num in split_time]
        return minutes * 60 + seconds

    def _time2str(self, seconds):
        minutes = str(seconds//60)
        seconds = str(seconds % 60)
        timestring = minutes + ':' + seconds
        return timestring

    def now(self):
        '''
        Returns : Time string
        '''
        return self._time_left(at_shop = True)

    def _time_taken(self, page_count, sleeptime):
        return page_count * sleeptime
    
    def _collate_data(self, page_count = 2, sleep_time = 1):
        time_remaining = self._time_left()
        if(time_remaining > sleep_time + 5):
            print('Started fetching page 1 at   {:>10}'.format(self._time2str(time_remaining)))
            current_page_dict = self._pull_pages(1, sleep_time)
        else:
            raise Exception('Not much time to fetch... Time Remaining : {:>10}s'.format(self._time2str(time_remaining)))
        if(page_count > 1 ):
            time_needed = self._time_taken(page_count-1, sleep_time)
            if (self._time_left() > time_needed + 10):
                self._goto_next_page(sleep_time * 0.5)
                print('Started fetching remaining pages ...')
                remaining_pages_dict = self._pull_pages(page_count-1, sleep_time)
                current_page_dict = self._merge_pages(current_page_dict, remaining_pages_dict)
            else:
                raise Exception('Not much time to fetch... Time Remaining : {:>10}s'.format(self._time2str(time_remaining)))
        return pd.DataFrame(current_page_dict)
                            
    def _run_2_win(self, page_count = 2, sleeptime = 1, entry_time = '2:52'):
        '''
        Entry time string must by in xx : xx format 
        
        '''
        time_to_fetch = page_count * sleeptime
        if not time_to_fetch > (self._time_left() - 2):
            while True:
        #         time_remaining = time_left()
                if(self._time_left() == self._str2time(entry_time)-2): 
                    print(f"Extracting data ... \nStarted fetching at {entry_time:>10} - 2 sec")
                    data_df = self._collate_data(page_count, sleeptime)
                    print("Extraction Complete...  Time elapsed : {} seconds".format(self._time_taken(page_count,sleeptime)))
                    return data_df
        else:
            raise Exception("More time needed...try reducing page count")
    
    def _quality(self, df):
        if(df.shape[0] == len(df.timestamp.unique())):
            return "nice"
        else:
            return "bad"

        
    def _num2color(self, df, nummap):
        df['num2color'] = df.number.map(nummap)
        return df
    
    
    def _fetch_pages(self, numrows, numentry = 10, sleeptime = 1):
        if numrows <= numentry:
            numpages = 2
        elif numrows % numentry == 0:
            numpages = numrows // numentry
        else:
            numpages = numrows // numentry + 1
        time_needed = self._time_taken(numpages, sleeptime) * 5
        if self._time_left() < (time_needed):
            raise Exception(f'Need more time to fetch... Try reducing datapoints... Time needed : {time_needed}s')
        print(f'Started fetching {numpages} pages .. at {self.now()}s')
        pages = pd.DataFrame(self._pull_pages(numpages, sleeptime))
        print('Fetch completed ... !')
        print('Data collected has ' + self._quality(pages) + ' quality !')
        return pages

    
    def get_gamedf_v2(self, n_datapoints, entry_time = '2:49', \
                      from_server = None, \
                      wait4next_bttn = 2,  \
                      sleeptime = 1, \
                      fetch_till = 5, \
                      fetch_from = '2:50'):
        counter_restarted = False
        if entry_time == '00:00':
            print('Waiting for counter to restart...')
            while self._time_left() == 0:
                pass
            entry_time_val = self._str2time(self.now()) - 5
            entry_time = self._time2str(entry_time_val) #2:45
            counter_restarted = True
        starttime = time.time()
        current_server = from_server
        if current_server is None:
            current_server = self.current_server
        print(f'Fetching from server : {self.servers[current_server - 1]} ! (Time now : {self.now()}s)')


        #strat fresh erase history of page traversal
        self.reload(wait4next_bttn)
        timeval = self._str2time(entry_time)
        timeval = timeval - wait4next_bttn
        entry_time = self._time2str(timeval)
        mintime = fetch_till
        maxtime = self._str2time(fetch_from)
        
        if not counter_restarted:
            entry_time_val = self._str2time(entry_time) - 2
            entry_time = self._time2str(entry_time_val)
        while True:
            if self._time_left() == self._str2time(entry_time):
                timerem = self._time_left()
                print(f'Entering at shop time {self._time2str(timerem)}...')
                if (timerem > mintime) & (timerem < maxtime):
                    #fetch pages based on ndata count - will throw error if time not sufficient
                    df = self._fetch_pages(n_datapoints, numentry = 10, sleeptime = sleeptime)

                    print('Processing data ...')
                    #problem with nadal ; first datapoint can be empty at times
                    if df.price[0] == 0:
                        df = df.tail(-1)
                    df['color'] = [self.nummap[num] for num in df['number']]
                    colorarray = df['color'].values 
                    df['colorseries'] = self.color_encoder.transform(colorarray.reshape(-1,1))

                    #feature crossing - additive
                    df['num_plus_color'] = df['number'] + df['colorseries']

                    #multiplicative
                    df['num_times_color'] = df['number'] * df['colorseries']
                    df.sort_index(ascending = False, inplace = True)
                    df.reset_index(drop = True, inplace = True)
                    print()
                    print('*' * 50)
                    print('Datapoints arrangement : Past to Future ...')
                    print('*' * 50)
                    print()
                    print('Data parsed successfully ...! ', end = '')
                    elapsedtime = time.time() - starttime
                    print(f'Time elapsed : {elapsedtime : > 1.2f} s')
                    return df
                else:
                    timerem = self._time2str(timerem)
                    raise Exception(f'Time outside fetching range.. Time Now : {timerem: >5}s')
                #if successful break the loop

                
    
        
    def hours2Ngames(self, t_minus, gametime_mins = 3):
        minutes = t_minus * 60
        n_games = minutes // 3
        return n_games
    
    def games2Nhours(self, games, gametime_mins = 3):
        minutes = games * gametime_mins
        hours = minutes // 60
        return hours


