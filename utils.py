import pandas as pd
# Various functions/utilities used to analalyze coronavirus tweets

def read_data():
    data = pd.read_csv('full_dataset-clean.tsv.gz', sep='\t', compression='gzip')
    return(data)