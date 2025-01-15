import csv
import os
import pandas as pd
from glob import glob
import json


def drone_df(target_folder):
    annotations_path = os.path.join(target_folder, 'annotations')
    images_path = os.path.join(target_folder, 'images')
    json_file = os.path.join(os.getcwd(), 'data_script/text_descriptions/gemini.json')

    with open(json_file, 'r') as f:
        descriptions = json.load(f)

    image_files = glob(os.path.join(images_path, '*.jpg'))
    image_dict = {os.path.splitext(os.path.basename(img))[0]: img for img in image_files}

    data = []
    for ant_file in glob(os.path.join(annotations_path, '*.txt')):
        base_filename = os.path.splitext(os.path.basename(ant_file))[0]
        if base_filename in image_dict:
            data.append({
                'image_path': image_dict[base_filename],
                'annotation_path': ant_file,
                'description': descriptions.get(base_filename + '.jpg', '')
            })

    df = pd.DataFrame(data)
    df['parsed_annotations'] = df['annotation_path'].apply(parse_annotations)

    return df


def parse_annotations(annotation_path):
    annotations = []
    with open(annotation_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            cleaned_line = [val for val in line if val.strip()]
            if len(cleaned_line) == 8:
                try:
                    annotations.append([int(val) for val in cleaned_line])
                except ValueError:
                    print(f"Non-integer value found in file {annotation_path}: {cleaned_line}")
                    continue
            else:
                print(f"Unexpected number of values in line in file {annotation_path}: {cleaned_line}")
    return annotations
