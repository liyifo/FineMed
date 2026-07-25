import os
import json

from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
import gensim
import dill
import pandas as pd



def segmentation_words(text, sections):
    section_indices = {}
    for section, names in sections.items():
        for name in names:
            index = index = text.find(name)
            if index != -1:
                section_indices[section] = (index, index + len(name))
                break


    section_indices = sorted(section_indices.items(), key=lambda e: e[1][0])
    section_indices.append(('', (len(text), 0)))
    note_sections = {}
    for i, (section, (start, end)) in enumerate(section_indices[:-1]):
        next_start = section_indices[i + 1][1][0]
        if next_start < end:
            note_sections[section] = []
        else:
            note_sections[section] = text[end:next_start]
    if len(note_sections) == 0:
        note_sections['others'] = text
    else:
        note_sections['others'] = []
    all_note_sections = {}
    for section in sections:
        all_note_sections[section] = note_sections[section] if section in note_sections else []
    return all_note_sections



def segmentation_dataset(dataset, section_titles):

    for index, sample in tqdm(dataset.iterrows()):
        text = sample['TEXT'].lower()
        # words = [word for word in tokenizer.tokenize(text.lower()) if not word.isnumeric()]
        # word_ids = [word2id.get(word, word2id['**UNK**']) for word in words]

        note_sections = segmentation_words(text, section_titles)
        dataset.at[index, 'sections'] = note_sections
    dill.dump(obj=dataset, file=open('data_final.pkl', 'wb'))





if __name__ == '__main__':
    section_titles = json.load(open('title_synonyms.json'))
    # print(section_titles)

    tokenizer = RegexpTokenizer(r'\w+')
    # word2id, id2word = load_vocab(os.path.join(MIMIC3_PATH, 'word2vec_sg0_100.model'))

    
    # dataset = json.load(open(os.path.join(EXTRACTED_PATH, f'mimic3.json'), encoding='utf-8'))
    dataset = dill.load(open('./data_final.pkl', "rb"))
    dataset['sections'] = None
    #dataset = pd.read_csv('./latest_records.csv')
    segmentation_dataset(dataset, section_titles)
