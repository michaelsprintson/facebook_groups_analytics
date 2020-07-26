import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from helper_functions import *

import nltk
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
[stopWords.add(i) for i in ['u', 'like', 'yeah', 'ur', 'get', 'i\'m']];

def message_count(total_messages, participant_names):
    return sort_dict({pn:len([i for i in total_messages if i['sender_name'] == pn]) for pn in participant_names})

def average_character_count(total_messages, participant_names):
    return sort_dict({pn:np.average([len(i['content']) for i in total_messages if i['sender_name'] == pn and 'content' in i.keys() ]) for pn in participant_names})

def total_character_count(total_messages, participant_names):
    return sort_dict({pn:sum([len(i['content']) for i in total_messages if i['sender_name'] == pn and 'content' in i.keys() ]) for pn in participant_names})
    
def most_reacted_to_message(total_messages_with_id):
    reacted_to_messages = {idx:i['reactions'] for idx, i in total_messages_with_id.items() if 'reactions' in i.keys()}
    maxid = sorted(reacted_to_messages, key = lambda i: len(reacted_to_messages[i]))[-1]
    return total_messages_with_id[maxid]

def reactions_for_participant(total_messages, participant):
    participant_messages = [i for i in total_messages if i['sender_name'] == participant]
    reactions_recieved_by_participant = [i['reactions'] if 'reactions' in i.keys() else []for i in participant_messages]
    return flat_list_n(reactions_recieved_by_participant)

def reaction_calc(total_reaction_list):
    subparticipant_reactions_func = lambda reactions, actor: sort_dict(dict(Counter([i['reaction'] for i in reactions if i['actor'] == actor])))
    reaction_number_by_participant = sort_dict(dict(Counter([i['actor'] for i in total_reaction_list])))
    reactions_stats = {k:{'number':v, 'reacts': {react_encoding_to_name[ki]:vi for ki,vi in subparticipant_reactions_func(total_reaction_list, k).items()}} for k,v in reaction_number_by_participant.items()}
    return reactions_stats

def reactions_recieved_by_all(participant_names, total_messages, returntype = 'dict'):
    react_stats = {}
    for pname in participant_names:
        f_reactions_recieved_by_participant = reactions_for_participant(total_messages, pname)
        reaction_stats_for_participant = reaction_calc(f_reactions_recieved_by_participant)
        to_sum = [v['reacts'] for k,v in reaction_stats_for_participant.items()]
        sd = dict(pd.DataFrame(to_sum).sum())
        react_stats[pname] = {'total': sum(sd.values()), 'stats':sort_dict(sd)}
    if returntype == 'dict':
        return react_stats
    return pd.DataFrame(react_stats).T.sort_values('total', ascending = False)

def react_to_messages_ratio(total_messages, participant_names):
    allreacts = list(react_encoding_to_name.values())
    messagecounts = message_count(total_messages, participant_names)
    react_stats = reactions_recieved_by_all(participant_names, total_messages, returntype='dict')

    react_message_df = pd.DataFrame({k:{'total reacted tos':sum(v['stats'][i] for i in allreacts if i in v['stats']), 'total messages': messagecounts[k]} for k,v in react_stats.items()}).T
    react_message_df['reacted_to_message_ratio'] = react_message_df['total reacted tos'] / react_message_df['total messages']
    react_message_df.sort_values('reacted_to_message_ratio', ascending=False)
    return react_message_df

def reactions_recieved_by_participant(total_messages, participant):
    f_reactions_recieved_by_participant = reactions_for_participant(total_messages, participant)
    return pd.DataFrame(reaction_calc(f_reactions_recieved_by_participant)).T

def reactions_given_by_all(total_messages):
    reacted_to_messages = flat_list([i['reactions'] for i in total_messages if 'reactions' in i])
    return pd.DataFrame(reaction_calc(reacted_to_messages)).T
    
def average_and_max_reactions_for_participant(total_messages, participant): ##BROKEN BUT I DONT KNOW WHY
    reacted_to_messages = flat_list([i['reactions'] for i in total_messages if i['sender_name'] == participant and 'reactions' in i])
    return np.average([len(i) for i in reacted_to_messages]), max([len(i) for i in reacted_to_messages])

def self_reacts(total_messages, participant):
    participant_messages = [i for i in total_messages if i['sender_name'] == participant] 
    return messages_reactions_given_by_participant(participant_messages, participant)

def messages_reactions_given_by_participant(total_messages, participant):
    return [i for i in total_messages if 'reactions' in i.keys() and len([j for j in i['reactions'] if j['actor'] == participant]) > 0]

def reactions_given_by_participant(total_messages, participant):
    participant_messages = messages_reactions_given_by_participant(total_messages, participant)
    return reactions_given_by_all(participant_messages)

def filter_word_list(si):
    sinoempty = [i for i in si if i != '']
    postags = nltk.pos_tag(sinoempty)
    return [i[0] for i in postags if i[1] == 'NN']
def get_word_list(s):
    return [si for si in filter_word_list(s.lower().split(' ')) if si not in stopWords]
def most_common_words(mes):
    return pd.Series(Counter(flat_list([get_word_list(s) for s in mes]))).sort_values(ascending = False)

def most_common_words_by_participant(total_messages, participant):
    part_messages = [i['content'] for i in total_messages if i['sender_name'] == participant and 'content' in i]
    return most_common_words(part_messages)

def nickname_changes_and_times(total_messages, participant):
    pnamedict = defaultdict(list)
    ptimedict = {}
    name_changes = [i for i in total_messages if 'content' in i and 'set the nickname for' in i['content']]
    for i in name_changes:
        if participant in i['content']:
            pnamedict[i['sender_name']].append(i['content'].split(participant)[1].strip(' to'))
            ptimedict[i['content'].split(participant)[1].strip(' to')] = i['timestamp_ms']
    pnamereturn = pd.DataFrame(pd.Series(dict(pnamedict)))
    
    ptimedict = sort_dict(ptimedict, reverse = False)
    pdl = list(ptimedict.values())
    pdk = list(ptimedict.keys())
    namezip = list(zip(pdl[0:-1], pdl[1:]))
    times_per_name = [i[1] - i[0] for i in namezip]
    ptimereturn = pd.Series(sort_dict({pdk[idx]:times_per_name[idx] for idx in range(len(pdk)-1)}))

    return pnamereturn, ptimereturn

def group_name_changes_timeline(total_messages):
    group_name_changes = [i for i in total_messages if 'content' in i and 'named the group' in i['content']]
    names_to_times = [i['timestamp_ms'] for i in group_name_changes]
    namezip = list(zip(names_to_times[0:-1], names_to_times[1:]))
    times_per_name = [i[1] - i[0] for i in namezip]
    group_name_changes = sort_dict({group_name_changes[idx]['content'].split('named the group ')[1]:times_per_name[idx] for idx in range(len(group_name_changes)-1)})
    return group_name_changes

def usage_participant_highlighted(total_messages, participant_names, specialpn):

    fig = plt.figure(figsize = (20,20))
    for pn in participant_names:
        stamps_per_person = [i['timestamp_ms'] for i in total_messages if i['sender_name'] == pn]
        hist = function_hist(stamps_per_person)[0]
        if pn == specialpn:
            plt.plot(hist, label = pn, linewidth = 5)
        else:
            plt.plot(hist, label = pn, linewidth = 0.4)
    plt.legend()
    plt.savefig(f'pictures/{specialpn}_specific.png')

def word_usage_highlighted(total_messages, participant_names, wordiq):
    scam_stamps = [i['timestamp_ms'] for i in total_messages if 'content' in i and wordiq in i['content']]
    fig = plt.figure(figsize = (20,20))
    for pn in participant_names:
        stamps_per_person = [i['timestamp_ms'] for i in total_messages if i['sender_name'] == pn]
        hist = function_hist(stamps_per_person)[0]
        plt.plot(hist, label = pn, linewidth = 0.4)
    hist = function_hist(scam_stamps, special = True)[0]
    plt.plot(hist, label = wordiq, linewidth = 5)
    plt.legend()
    plt.savefig(f'pictures/{wordiq}_specific.png')

def most_common_photo_sender(total_messages):
    return pd.Series(sort_dict(dict(Counter([i['sender_name'] for i in total_messages if 'photos' in i]))))

def tag_correlation_matrix(participant_names, total_messages):
    participantmatrix = np.zeros((len(participant_names), len(participant_names)))
    for i in range(len(participant_names)):
        for j in range(len(participant_names)):
            participant = participant_names[i]
            participant2 = participant_names[j]
            part_sent_messages = [i['content'] for i in total_messages if i['sender_name'] == participant and 'content' in i and f"@{participant2}" in i['content']]
            participantmatrix[i][j] = len(part_sent_messages)

    f, ax = plt.subplots(figsize=(11, 10))
    f.set_facecolor((1, 1, 1))
    # cmap = sns.diverging_palette(0, max(flat_list(participantmatrix)), as_cmap=True)
    sns.heatmap(participantmatrix, square = True, annot = True, center = 0, vmin = 0, vmax = 50, cmap= 'coolwarm',  xticklabels=participant_names,yticklabels=participant_names )
    plt.savefig(f'pictures/corrmatrix.png')