from gpt4all import GPT4All
import PyPDF2
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_sm")

def add_embedding_and_cosine_similarity(df, fixed_text):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df['content_embedding'] = df['content'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    fixed_text_embedding = model.encode(fixed_text, convert_to_tensor=True)
    df['similarity'] = df['content_embedding'].apply(lambda x: cosine_similarity([x.cpu().numpy()], [fixed_text_embedding.cpu().numpy()])[0][0])
    return df

def select_top_neighbors(df):
    df = df.sort_values('similarity', ascending=False)
    top_neighbors = df.head(30) # 原来是 10，现在改成 30。选更多段落
    return top_neighbors


def save_df_to_text(df_filtered, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index, row in df_filtered.iterrows():
            file.write(row['content'] + '\n\n')  # Writing content and a blank line

def process_text_file_for_embedding(file_path):
    # read and process text
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        current_segment = []
        segments = []
        for line in lines:
            if line.strip():
                current_segment.append(line.strip())
            else:
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []
        if current_segment:
            segments.append(' '.join(current_segment))

        df = pd.DataFrame(segments, columns=['content'])

        fixed_text = (
        "flow chemistry, continuous flow, process development, residence time, "
        "flow rate, mL/min, µL/min, temperature °C, catalyst, reagent feed, "
        "reactor type, tubular reactor, microreactor, microchannel, coil, "
        "inner diameter ID, mm, μm, conversion %, yield %, selectivity %, "
        "optimization, space time yield, mixer, back pressure regulator"
        )
        df_with_embeddings = add_embedding_and_cosine_similarity(df, fixed_text)
        df_top_neighbors = select_top_neighbors(df_with_embeddings)

        base_name = os.path.basename(file_path)
        new_name = 'Embedding_' + base_name.replace('Processed_', '')
        output_file_path = os.path.join(os.path.dirname(file_path), new_name)
        save_df_to_text(df_top_neighbors, output_file_path)

    return output_file_path
