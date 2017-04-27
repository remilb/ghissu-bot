import re, os, argparse
import pandas as pd

def read_tv_script(tv_script_dir, df):
    regex = re.compile(r"(  )?(?P<speaker>[^:]+)(?P<action>\([^:]+\))?:(?P<utterance>.+)$")
    i  = 0
    for (dirpath, dirnames, filenames) in os.walk(tv_script_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.sep.join([dirpath, filename])
                with open(file_path) as file:
                    print(file_path)
                    episode_name  =  filename.strip(".txt")
                    for line in file:
                        iterator = regex.finditer(line)
                        for m in iterator:
                            speaker = m.group('speaker').strip()
                            if speaker.lower().startswith("scene") or speaker.lower().startswith("source"):
                                continue

                            utterance = m.group('utterance').strip()

                            action = ""
                            if m.group('action') is not None:
                                action = m.group('action')

                            if "[" in speaker:
                                continue
                            #action = action.strip(")").rstrip("(")
                            utterance = re.sub(r'\([^)]*\)', '', utterance).replace('"', '').replace('"', '')

                            df.loc[i] = [speaker,  utterance, int(i-1), episode_name]
                            i +=1
                            print(speaker, action, utterance)
    return df
                            #print("we are here"),
                            # TODO, use the utterance...

if __name__ == "__main__":
    # python3 read_tv_scripts.py -d data/tv_scripts/
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--in_dname", help="read directory that holds all the tv scripts", type=str, required=True)
    args = parser.parse_args()
    headers = [ 'speaker', 'utterance', 'prevId', 'episode_name']
    df = pd.DataFrame(data=None,index=None, columns=headers);
    tv_script_dir = args.in_dname
    df = read_tv_script(tv_script_dir, df)
    df.to_csv("sample.csv")
