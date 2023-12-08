# import json 
# from PIL import Image
# import os

# path = "/workspace/ailab/phucnd/TransWeather/data_noise/annotation.json"

# path_train_save = "/workspace/ailab/phucnd/TransWeather/data/train/input"
# path_test_save = "/workspace/ailab/phucnd/TransWeather/data/test/input"
# path_root_image = "/workspace/ailab/phucnd/TransWeather/data_noise/images"

# a = open(path)
# data = json.load(a)

# image_train_id = data['train']
# image_test_id = data['val']

# for i in data['images']:
#     if i['id'] in data['train']:
#         if (i['type'] != "rain"):
#             path_image = os.path.join(path_root_image, i['filename'])
#             with Image.open(path_image) as img:
#                 img.save(os.path.join(path_train_save,f'{i["filename"]}'))
#     else :
#         if (i['type'] != "rain"):
#             path_image = os.path.join(path_root_image, i['filename'])
#             with Image.open(path_image) as img:
#                 img.save(os.path.join(path_test_save,f'{i["filename"]}'))

import os

def write_links_to_file(folder_path, output_file):
    with open(output_file, 'w') as file:
        for root, dirs, files in os.walk(folder_path):
            for name in files + dirs:
                item_path = os.path.join(root, name)
                file.write(item_path + '\n')

folder_path = '/workspace/ailab/phucnd/TransWeather/data/test/input'

output_file = 'output124.txt'

write_links_to_file(folder_path, output_file)



