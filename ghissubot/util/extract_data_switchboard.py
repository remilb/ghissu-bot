import glob
#from swda import Transcript
from preprocess_textacy import *
import os

def group_tags(tag):
    """
    Group 40 tags to a predefined set of 10 tags
    
    4 aa, 14 bk, 21 na, 39 aap_am, 6 ba, 9 ny, 29 ng (accept) -> A
    1 sd, 13 nn, (non opinionated) -> NO
    2 b, 5 %, 11 %, 38 bd (bad short generic) -> BSG
    3 sv, 15 h (opinionated) -> O
    7 qy, 12 qw, 16 qy^d, 18 bh, 22 ad, 25 qo, 26 qh, 40 ^g, 41 qw^d - (questions) -> Q
    24 b^m, 23 ^2, 20 bf (summarize, repeat) -> S
    28 ar, 27 ^h, 30 br, 34 arp_nd (reject, non understanding) -> R
    10 fc, 31 no, 32 fp, 33 qrr, 35 t3, 36 oo_co_cc, 37 t1, 42 fa, 43 ft (conventional responses, generic) -> C
    8 x (non verbal) -> NV
    17 fo_o_fw_by_bc, 19 ^q, 22 ad (other , quotes, add on) -> OT
    """

    if tag in ('aa', 'bk', 'na', 'aap_am', 'ba', 'ny', 'ng'):
        tag = 'A' #Accept
    elif tag in ('sd', 'nn'):
        tag = 'NO' #No objective
    elif tag in ('b', '%', '+', 'bd'):
        tag = 'BSG' #Bad short generic
    elif tag in ('sv', 'h'):
        tag = 'O' #Opinionated
    elif tag in ('qy', 'qw', 'qy^d', 'bh', 'ad', 'qo', 'qh', '^g', 'qw^d'):
        tag = 'Q' #Question
    elif tag in ('b^m', '^2', 'bf'):
        tag = 'S' # Summarize, repeat
    elif tag in ('ar', '^h', 'br', 'arp_nd'):
        tag = 'R' # Reject
    elif tag in ('fc', 'no', 'fp', 'qrr', 't3', 'oo_co_cc', 't1', 'fa', 'ft'):
        tag = 'C' # Conventional response
    elif tag in ('x'):
        tag = 'NV' # Non verbal
    elif tag in ('fo_o_fw_"_by_bc', '^q', 'ad'):
        tag = 'OT' #Others
    else:
        print("Unknown tag", tag)
        assert False
    return tag
    
def damsl_act_tag(act_tag):
    """
    Seeks to duplicate the tag simplification described at the
    Coders' Manual: http://www.stanford.edu/~jurafsky/ws97/manual.august1.html
    """
    d_tags = []
    tags = re.split(r"\s*[,;]\s*", act_tag)
    for tag in tags:
        if tag in ('qy^d', 'qw^d', 'b^m'):
            pass
        elif tag == 'nn^e':
            tag = 'ng'
        elif tag == 'ny^e':
            tag = 'na'
        else:
            tag = re.sub(r'(.)\^.*', r'\1', tag)
            tag = re.sub(r'[\(\)@*]', '', tag)
            if tag in ('qr', 'qy'):
                tag = 'qy'
            elif tag in ('fe', 'ba'):
                tag = 'ba'
            elif tag in ('oo', 'co', 'cc'):
                tag = 'oo_co_cc'
            elif tag in ('fx', 'sv'):
                tag = 'sv'
            elif tag in ('aap', 'am'):
                tag = 'aap_am'
            elif tag in ('arp', 'nd'):
                tag = 'arp_nd'
            elif tag in ('fo', 'o', 'fw', '"', 'by', 'bc'):
                tag = 'fo_o_fw_"_by_bc'
        d_tags.append(tag)
    # Dan J says (p.c.) that it makes sense to take the first;
    # there are only a handful of examples with 2 tags here.
    return d_tags[0]

metadata_file = 'swda/swda-metadata.csv'
data_dir = "/home/sharath/Downloads/swda/swda/"
folders = glob.glob(data_dir + "*")
print(folders)
flag = True
prev_dir = os.getcwd()
os.chdir(data_dir)
for folder in folders:
    files = glob.glob(folder + "/*")
    for filename in files:
        if flag:
            print(folder + " ----------- " + filename)
            data = pd.read_csv(filename)
            flag = False
        else:
            #frame = [tweet_df, tweet_df2]
            print(folder + " ----------- " + filename)
            data = pd.concat([data, pd.read_csv(filename)])
os.chdir(prev_dir)
data['act_tag_new'] = data['act_tag'].apply(damsl_act_tag).apply(group_tags)
#my_set = set(data['act_tag_new'])
#print(len(my_set))
#print(my_set)
#assert False
data.to_csv("all_swbd_data.csv")
preprocess_cnn(data, data_col='text', label_col='act_tag_new', data_filename='swbd_utterance.csv',label_filename='swbd_act.csv' )


