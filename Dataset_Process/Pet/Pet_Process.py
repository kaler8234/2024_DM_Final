import shutil
import os

origin_dataset_file_prefix = "dataset/Pet/Pet/"
output_dataset_file_prefix = "dataset/Pet/Pet_byclass/"


def txt_to_dict(filename):
    dic = {}

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.startswith("#"):
                parts = line.split(' ')
                dic[parts[0] + '.jpg'] = parts[1]

    return dic


def make_pic_into_folder(annotation_dic):
    pics = os.listdir(origin_dataset_file_prefix + 'images')
    for pic in pics:
        if not pic.endswith('.mat') and pic in annotation_dic:
            id = annotation_dic[pic]
            if not os.path.exists(output_dataset_file_prefix + id):
                os.makedirs(output_dataset_file_prefix + id)

            shutil.copy(origin_dataset_file_prefix + 'images/' + pic, output_dataset_file_prefix + id + "/" + pic)


if __name__ == '__main__':
    annotation_dic = txt_to_dict(origin_dataset_file_prefix + "annotations/list.txt")
    make_pic_into_folder(annotation_dic)
