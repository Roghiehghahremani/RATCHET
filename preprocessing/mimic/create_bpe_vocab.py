import pandas as pd # pandas is used to read and process the dataset (mimic_cxr_labeled.csv).

"""yteLevelBPETokenizer is a subword tokenizer from the tokenizers library, which efficiently handles rare and 
frequent words in text data."""
from tokenizers import ByteLevelBPETokenizer 

"""Reads the MIMIC-CXR labeled dataset from a CSV file.
Drops missing/empty reports to ensure only valid text data is processed.
Converts the reports into a Python list (reports), which will be used for training the tokenizer."""
MIMIC_REPORTS = './mimic_cxr_labeled.csv'

reports = pd.read_csv(MIMIC_REPORTS)
reports = reports.dropna(subset=['Reports'])  # delete empty reports
reports = list(reports['Reports'].values)

"""Writes all the reports line by line to /tmp/mimic.txt.
This file serves as input for training the tokenizer."""
with open('/tmp/mimic.txt', 'w') as f:
    for item in reports:
        f.write("%s\n" % item)

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files='/tmp/mimic.txt', vocab_size=20000, min_frequency=2, special_tokens=[
    '<pad>',
    '<s>',
    '</s>',
    '<unk>',
    '<mask>',
])

tokenizer.save('.', 'mimic')
