import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import six
from collections import Counter

from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import plotly.offline as pyo
pyo.init_notebook_mode()

check_reacts_and_translate = lambda m: {i['actor']:react_encoding_to_name[i['reaction']] for i in m['reactions']} if 'reactions' in m.keys() else {}
sort_and_cut_dict = lambda d, cut=5: {i:d[i] for i in sorted(d, key = lambda i: d[i], reverse=True)[:cut]}
ms_to_day = lambda t: int(np.around(t / (8.64 * 10**7), decimals = 0))

react_encoding_to_name = defaultdict(lambda: "<custom emote>", {'ð\x9f\x91\x8d': 'thumbs up', 'ð\x9f\x92\x97':'heart', 'ð\x9f\x98¢':'cry', 'ð\x9f\x98®':'wow', 'ð\x9f\x98\xa0':'angry', 'â\x9d¤':'heart', 'ð\x9f\x98\x86':'laughing', 'ð\x9f\x91\x8e': 'thumbs down', 'ð\x9f\x98\x8d': 'heart eyes', 
"\u00e2\u009d\u0093": 'question mark', "ð\x9f¤\xa0": "cowboy hat man", 'ð\x9f\x91\x8dð\x9f\x8f¼': 'thumbs up', 'ð\x9f¤£': "laughing", 'ð\x9f§\x90': 'thinking', 'ð\x9f\x98¬': "teeth grit", 'ð\x9f¤®': 'barf', 'ð\x9f¤¨': "thinking",
"ð\x9f\x91\x8eð\x9f\x8f¼": "thumbs down", 'ð\x9f\x91\x80': "eyes", 'ð\x9f\x98\x82': "laughing", 'ð\x9f\x92\x96': 'heart', 'ð\x9f\x98°': "sweaty brow", 'ð\x9f\x93¸':"camera"})


def get_participant_names(mesasgefiles):
    participants = mesasgefiles[0]['participants']
    participant_names = [i['name'] for i in participants]
    return participant_names

def get_messages(messageloc = 'facebookdata/FredySupportGroup_ibkzDAfRNA copy/'):
    messagefilenames = [i for i in os.listdir(messageloc) if i[-5:] == '.json']
    mesasgefiles = [json.load(open(f'{messageloc}{i}', 'r')) for i in messagefilenames]
    participant_names = get_participant_names(mesasgefiles)
    total_messages = flat_list([i['messages'] for i in mesasgefiles])
    total_messages = sorted(total_messages, key = lambda i: i['timestamp_ms'])
    total_messages_with_id = {i:total_messages[i] for i in range(len(total_messages))}
    return total_messages, total_messages_with_id, participant_names

class messages:
    def __init__(self, location = 'facebookdata/messages/inbox/s_ibkzdafrna/'):
        # to process messages into dataframe, create mappings for each user and each emote used
        self.total_messages, self.total_messages_with_id, self.participant_names = get_messages(messageloc = location)
        self.mdf = pd.DataFrame(self.total_messages).reset_index()
        self.userfilters = {i:self.mdf['sender_name'] == i for i in self.participant_names}

        reacted_messges = self.mdf[~self.mdf['reactions'].isna()]
        self.person_reacting_dict = defaultdict(lambda: {})
        self.mdf['reacts'] = reacted_messges.apply(self.process_message, axis = 1)
        self.emote_df = pd.DataFrame.from_dict(self.person_reacting_dict, orient = 'index')


    def get_user_messages(self, username):
        return self.mdf[self.userfilters[username]]
    
    def process_message(self, rm):
        rnt = check_reacts_and_translate(rm)
        for p,e in rnt.items():
            self.person_reacting_dict[rm['index']] = {'reactor': p, 'reacted': rm['sender_name'], 'emote': e}
        return rnt

    def filter_df(self, selector = 'emote', filter = 'laughing', from_filt = None):
        # selector can be emote, reactor, or reacted
        # filter can be any emote name or chat member name, or for selector "emote" you can do "all"
        
        if from_filt is None:
            if 'emote' == selector and filter == 'all':
                return self.mdf[~self.mdf['reactions'].isna()]
            return self.mdf.loc[self.emote_df[self.emote_df[selector] == filter].index]
        else:
            if 'emote' == selector and filter == 'all':
                return from_filt[~from_filt['reactions'].isna()]
            idxs = set(self.emote_df[self.emote_df[selector] == filter].index.to_numpy()).intersection(set(from_filt.index.to_numpy()))
            return from_filt.loc[idxs]

    def get_most_reacted_message(self, from_filt = None):
        if from_filt is None:
            return self.mdf.loc[self.filter_df('emote','all').apply(lambda r: sum(Counter(r['reacts'].values()).values()),axis = 1).sort_values(ascending = False).iloc[:5].index]
        return from_filt.loc[self.filter_df('emote','all', from_filt).apply(lambda r: sum(Counter(r['reacts'].values()).values()),axis = 1).sort_values(ascending = False).iloc[:5].index]


    def get_most_reacted_one_emotion(self, emotion, from_filt = None):
        if from_filt is None:
            return self.mdf.loc[self.filter_df('emote', emotion).apply(lambda r: Counter(r['reacts'].values())[emotion],axis = 1).sort_values(ascending = False).iloc[:5].index]
        return from_filt.loc[self.filter_df('emote', emotion, from_filt).apply(lambda r: Counter(r['reacts'].values())[emotion],axis = 1).sort_values(ascending = False).iloc[:5].index]

    def get_most_messages(self, from_filt = None, cut = 5):
        if from_filt is None:
            name_to_num = dict(Counter(self.mdf['sender_name'].values))
        else:
            name_to_num = dict(Counter(from_filt['sender_name'].values))
        return sort_and_cut_dict(name_to_num, cut)
    
    def get_avg_char_count(self, cut = 5):
        len_dict = {}
        for p in self.participant_names:
            um = self.get_user_messages(p)
            len_dict[p] = np.around(np.average(um[~um['content'].isna()]['content'].apply(lambda x: len(x))), decimals = 2)
        return sort_and_cut_dict(len_dict, cut)
    
    def get_react_to_messages_ratio(self, cut = 5):
        ratio_dict = {}
        for p in self.participant_names:
            ratio_dict[p] = np.around(len(self.emote_df[self.emote_df['reacted'] == p]) / len(self.get_user_messages(p)), decimals = 2)
        return sort_and_cut_dict(ratio_dict, cut = cut)

    def get_emotes_to_messages_ratio(self):
        ratio_dict = defaultdict(lambda: {})
        for p in self.participant_names:
            pemote = self.emote_df[self.emote_df['reacted'] == p]
            for e in list(react_encoding_to_name.values()):
                ratio_dict[p][e] = len(pemote[pemote['emote'] == e]) / len(self.get_user_messages(p))
        return pd.DataFrame(ratio_dict)
    
    def get_all_reactions_recieved(self, pname):
        f = self.emote_df[self.emote_df['reacted'] == pname][['emote','reactor']].to_dict(orient = 'records')
        react_dict = defaultdict(lambda: defaultdict(int))
        for fi in f:
            react_dict[fi['reactor']][fi['emote']] += 1
        return pd.DataFrame.from_dict(react_dict)

    def get_all_reactions_given(self, pname):
        f = self.emote_df[self.emote_df['reactor'] == pname][['emote','reacted']].to_dict(orient = 'records')
        react_dict = defaultdict(lambda: defaultdict(int))
        for fi in f:
            react_dict[fi['reacted']][fi['emote']] += 1
        return pd.DataFrame.from_dict(react_dict)
    
    def render_mpl_table(self, df, x_axis = "", y_axis = ""):
        fig =  ff.create_table(df.reset_index())
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            autosize = False,
            width= 200 * len(self.participant_names),
            height=500,
        )
        return fig
        # fig.write_image("table_plotly.png", scale=2)
        # fig.show()

    def self_reacts(self, pname):
        udf = self.emote_df[self.emote_df['reactor'] == pname]
        sdf = udf[udf['reactor'] == udf['reacted']]
        selfreactdf = self.mdf.loc[sdf.index][['sender_name', 'content', 'reacts']]
        selfreactdf['self-react'] = sdf['emote']
        return selfreactdf



def nickname_changes_and_times(total_messages, participant):
    pnamedict = defaultdict(list)
    ptimedict = {}
    name_changes = [i for i in total_messages if 'content' in i and 'set the nickname for' in i['content']]
    for i in name_changes:
        if participant in i['content']:
            pnamedict[i['sender_name']].append(i['content'].split(participant)[1].strip(' to'))
            ptimedict[i['content'].split(participant)[1].strip(' to')] = i['timestamp_ms']
    # pnamereturn = dict(pnamedict)
    
    ptimedict = sort_dict(ptimedict, reverse = False)
    pdl = list(ptimedict.values())
    pdk = list(ptimedict.keys())
    namezip = list(zip(pdl[0:-1], pdl[1:]))
    times_per_name = [ms_to_day(i[1] - i[0]) for i in namezip]
    ptimereturn = pd.Series(sort_dict({pdk[idx]:times_per_name[idx] for idx in range(len(pdk)-1)}))

    return sort_and_cut_dict(ptimereturn.to_dict(), 15)

def group_name_changes_timeline(total_messages):
    group_name_changes = [i for i in total_messages if 'content' in i and 'named the group' in i['content']]
    names_to_times = [ms_to_day(i['timestamp_ms']) for i in group_name_changes]
    namezip = list(zip(names_to_times[0:-1], names_to_times[1:]))
    times_per_name = [i[1] - i[0] for i in namezip]
    group_name_changes = sort_and_cut_dict({group_name_changes[idx]['content'].split('named the group ')[1]:times_per_name[idx] for idx in range(len(group_name_changes)-1)},15)
    return group_name_changes

def plot_dict(d, reverse = False, title = "",xaxis = ""):
    x_data = [i for i,j in d.items()]
    y_data = list(d.values())
    if reverse:
        x_data.reverse()
        y_data.reverse()
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))
        
        .add_xaxis(x_data)
        
        .add_yaxis(
            title,
            list(y_data),
            yaxis_index=0,
            label_opts= opts.LabelOpts(position="inside", rotate= 0, is_show = True, formatter='{b}')
        )
        .reversal_axis()

        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name=xaxis, position="top", offset=10),
            yaxis_opts=opts.AxisOpts(name="nickname", position="right", offset=10, is_show=False),

        )
        
    )
    return bar

def plot_overall_stats(df, reverse = False, title = "",xaxis = ""):
    bar = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))
    .add_xaxis([i.split(" ")[0] for i in list(df.index)])
    .add_yaxis("Message Count", list(df.mostmessages), yaxis_index = 0)
    .add_yaxis("Average Character Count", list(df.avgcharcount), yaxis_index = 1)
    .add_yaxis("Reacts to Messages Ratio", list(df.reacttm), yaxis_index = 2)
    #  .reversal_axis()
    .set_series_opts(label_opts=opts.LabelOpts(position="top", rotate= 30))
    .extend_axis(yaxis=opts.AxisOpts(name="Average Character Count", type_="value", position="left"))
    #  .extend_axis(xaxis=opts.AxisOpts(name="Name", type_="value", position="left"), xaxis_opts )
     .extend_axis(yaxis=opts.AxisOpts(type_="value", name="        Ratio",offset = 60, position="outside", max_=1))
    .set_global_opts(title_opts=opts.TitleOpts(title="Message Analytics for Elec Gang"),yaxis_opts=opts.AxisOpts(name="Message Count", position="right", offset=0), xaxis_opts=opts.AxisOpts(name="Name", position="bottom", offset=10, axislabel_opts = opts.LabelOpts(position="bottom", rotate= 30)))
    )
    return bar
    # bar.render_notebook()


































flat_list = lambda x: [i for l in x for i in l]
flat_list_n = lambda x: [i for l in x if type(l) in [list, np.ndarray] and len(l) > 0 for i in l ]

def sort_dict(input_dict, maxkeys = None, reverse = True):
    sorted_keys = sorted(input_dict, key = lambda x: input_dict[x], reverse = reverse)[:maxkeys]
    sorted_dict = {k:input_dict[k] for k in sorted_keys}
    return sorted_dict

def addkey(d, key, val):
    r = dict(d)
    r[key] = val
    return r





def function_hist(a, bins = 100, special = False):

    # 12 bins
    bins = np.linspace(a[0], a[-1], bins)
    if special:
        weightsa = np.ones_like(a) * 10
    else:
        weightsa = np.ones_like(a)
    # weightsa  = weightsa / sum(weightsa)
    hist = np.histogram(np.array(a), bins, weights = weightsa)
    return hist

def dict_to_cols(inputpd, dictcol):
    convertdict = {k:defaultdict(int,v) for k,v in inputpd[dictcol].to_dict().items()}
    newpd = inputpd.drop(dictcol, axis = 1).T.to_dict()
    [newpd[k].update(v) for k,v in convertdict.items()];
    return pd.DataFrame(newpd).T

import plotly.figure_factory as ff
import pandas as pd





#### old searching for reactions

# m_r = [i for i in meme_total_messages if i['sender_name'] == p and 'reactions' in i.keys() and 'content' in i.keys()]
# check_message = lambda m: sum([react_encoding_to_name[j['reaction']] == '<custom emote>' for j in m['reactions']])

# message_idx = 0
# flag = False
# while flag == False:
#     message_idx -= 1
#     flag = check_message(m_r[message_idx])
# print(m_r[message_idx])

# test = ([i['reaction'] for i in m_r[message_idx]['reactions']])
# test

# ([react_encoding_to_name[i['reaction']] for i in m_r[message_idx]['reactions']])