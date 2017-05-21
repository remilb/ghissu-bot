import glob
#from swda import Transcript
from ghissubot.util.preprocess_textacy import *
import os

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
data_dir = "/Users/shubhi/Documents/Spring 17/Masters Project/swda/"
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
            data = pd.concat([data, pd.read_csv(filename)])
os.chdir(prev_dir)
data['act_tag_new'] = data['act_tag'].apply(damsl_act_tag)
data.to_csv("all_swbd_data.csv")
preprocess_cnn(data, data_col='text', label_col='act_tag_new', data_filename='swbd_utterance.csv',label_filename='swbd_act.csv' )


