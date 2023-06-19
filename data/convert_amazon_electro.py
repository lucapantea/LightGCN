import json

def convert_json_to_text(json_file_path, text_file_path):
    with open(json_file_path, 'r') as json_file, open(text_file_path, 'w') as text_file:
        data = json.load(json_file)
        for i, sublist in enumerate(data):
            text_file.write(f"{i} ")
            text_file.write(" ".join(map(str, sublist)))
            text_file.write("\n")

# Converting the needed files
paths_to_convert = {'amazon-electro/train_data.json': 'amazon-electro/train.txt',
                    'amazon-electro/test_data.json': 'amazon-electro/test.txt',
                    'movielens/train_data.json': 'movielens/train.txt',
                    'movielens/test_data.json': 'movielens/test.txt',
                    }

for input_path, output_path in paths_to_convert.items():
    convert_json_to_text(input_path, output_path)
