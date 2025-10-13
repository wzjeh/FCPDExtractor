import pandas as pd
import re
import os

def preprocess_and_filter_text(df):
    pattern = r'(?i)\b(Acknowledgements|Acknowledgement|Data availability|ASSOCIATED CONTENT|Conflict of interest|Conflicts of interest|Conflict of interests|Conflicts of interests)\b' #Received:November2 for303
    stop_processing = False
    last_valid_index = None

    for index, row in df.iterrows():
        if stop_processing:
            break
        if re.search(pattern, row['content']):
            stop_processing = True
            cleaned_text = re.sub(pattern + r'[\s\S]*', '', row['content'])
            df.at[index, 'content'] = cleaned_text
            last_valid_index = index
        else:
            last_valid_index = index
    if last_valid_index is not None:
        return df.iloc[:last_valid_index + 1]
    else:
        return df


def save_df_to_text(df, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index, row in df.iterrows():
            file.write(row['content'] + '\n\n')


def process_text_file_for_processing(file_path):
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
        df_filtered = preprocess_and_filter_text(df)
        filtered_count = len(df_filtered)

        base_name = os.path.basename(file_path)
        new_name = 'Processed_' + base_name
        output_file_path = os.path.join(os.path.dirname(file_path), new_name)
        save_df_to_text(df_filtered, output_file_path)


    return output_file_path,filtered_count
