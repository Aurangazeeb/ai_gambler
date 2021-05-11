_login_dict = dict(
        login_tab_xpath = '//*[@id="q-app"]/div/footer/div/div/a[3]',\
        notice_bttn_xpath = '/html/body/div[3]/div[2]/div/div[3]/button',\
        #Login button
        login_button_xpath = '//*[@id="q-app"]/div/div/div/div/form/div[2]/button', \
        username_xpath = '/html/body/div[1]/div/div/div/div/form/div[1]/label[1]/div/div/div[2]/input', \
        password_xpath = '/html/body/div[1]/div/div/div/div/form/div[1]/label[2]/div/div/div[2]/input',\
        score_tab_xpath = '//*[@id="q-app"]/div/footer/div/div/a[3]'
        )

_entry_names = ['timestamp','price', 'number','color']
_num_entry = len(_entry_names)
_table_base = ['//*[@id="q-app"]/div/div/div/div[3]/div[4]/div[1]/table/tbody/tr[{1}]/td[{0}]']
_table_bases = _table_base * _num_entry
_table_base_dict = dict(zip(_entry_names, _table_bases))

_page_base = dict(page_base = '//*[@id="q-app"]/div/div/div/div[3]/div[4]/div[2]/div/div/button[{}]')
_time_base = dict(time_base = '//*[@id="q-app"]/div/div/div/div[3]/div[2]/span[2]')

_server_names = ['parity','sapre','bcone','emerd']
_server_base = dict(server_base = '//*[@id="q-app"]/div/div/div/div[2]/div/div[{}]')

_rnum = list(filter(lambda x : (x % 2 == 0) & (x is not 0) , range(10))) 
_gnum = list(filter(lambda x : (x % 2 != 0) & (x is not 5), range(10)))

_gcol = ['green'] * len(_gnum)
_rcol = ['red'] * len(_rnum)
_gvcol = 'green + violet'
_rvcol = 'red + violet'

_gmap = dict(zip(_gnum, _gcol))
_rmap = dict(zip(_rnum, _rcol))
_vmap = dict(zip((5, 0), (_gvcol, _rvcol)))

_nummap = {**_gmap, **_rmap, **_vmap}

_pageconstants = (_login_dict, _table_base_dict, _page_base, _time_base, _nummap, _server_base, _server_names)