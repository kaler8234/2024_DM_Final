import shutil
import os

origin_dataset_file_prefix = "dataset/CelebA/CelebA/"
output_dataset_file_prefix = "dataset/CelebA/CelebA_byclass/"


def txt_to_dict(filename):
    dic = {}

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split(' ')
            dic[parts[0]] = parts[1].replace('\n', '')

    return dic


def make_pic_into_folder(annotation_dic):
    pics = os.listdir(origin_dataset_file_prefix + 'img_align_celeba')
    for pic in pics:
        id = annotation_dic[pic]
        if not os.path.exists(output_dataset_file_prefix + id):
            os.makedirs(output_dataset_file_prefix + id)

        shutil.copy(origin_dataset_file_prefix + 'img_align_celeba/' + pic, output_dataset_file_prefix + id + "/" + pic)


if __name__ == '__main__':
    try:
        os.mkdir(output_dataset_file_prefix)
    except:
        pass
    annotation_dic = txt_to_dict(origin_dataset_file_prefix + "identity_CelebA.txt")
    make_pic_into_folder(annotation_dic)
