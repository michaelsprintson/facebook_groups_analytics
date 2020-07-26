import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

flat_list = lambda x: [i for l in x for i in l]
flat_list_n = lambda x: [i for l in x if type(l) in [list, np.ndarray] and len(l) > 0 for i in l ]

def sort_dict(input_dict, maxkeys = None, reverse = True):
    sorted_keys = sorted(input_dict, key = lambda x: input_dict[x], reverse = reverse)[:maxkeys]
    sorted_dict = {k:input_dict[k] for k in sorted_keys}
    return sorted_dict


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

react_encoding_to_name = {'ð\x9f\x91\x8d': 'thumbs up', 'ð\x9f\x92\x97':'purple heart', 'ð\x9f\x98¢':'cry', 'ð\x9f\x98®':'wow', 'ð\x9f\x98\xa0':'angry', 'â\x9d¤':'pink heart', 'ð\x9f\x98\x86':'laughing', 'ð\x9f\x91\x8e': 'thumbs down', 'ð\x9f\x98\x8d': 'heart eyes'}

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