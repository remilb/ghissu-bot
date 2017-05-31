import pandas as pd
def len_of_text(text):
    return (len(text.split(" ")))

def analysis(filename):
    print()
    print("performing analysis for " , filename)
    b1 = pd.read_csv(filename, sep="\n", names =['output'])
    b1['length'] = b1['output'].apply(len_of_text)
    print("median ",pd.DataFrame.median(b1) )
    print("mean ",pd.DataFrame.mean(b1) )
    b1_words = " ".join(b1['output'])
    list_b1 = list(" ".join(b1['output']))
    print("list len ", len(list_b1))
    set_b1=set(list_b1)
    print("set len ", len(set_b1))
    print ("total unique words ", len(set_b1)/(1.0 * len(list_b1)) )


f1 = "/Users/shubhi/Public/CMPS296/sample_convos/greedy_b1/all_output.txt"
f2 ="/Users/shubhi/Public/CMPS296/sample_convos/greedy_b2/all_output.txt"
f3= "/Users/shubhi/Public/CMPS296/10_epochs_beam.txt"
f4="/Users/shubhi/Public/CMPS296/10_epochs_no_beam.txt"
f5="/Users/shubhi/Public/CMPS296/15_epochs_beam.txt"
f6="/Users/shubhi/Public/CMPS296/15_epochs_no_beam.txt"

all_files = [f1,f2,f3,f4,f5,f6]

for file in all_files:
    analysis(file)

