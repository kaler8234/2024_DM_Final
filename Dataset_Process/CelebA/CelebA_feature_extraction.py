import numpy as np
from PIL import Image, ImageFilter
import os
from transformers import CLIPProcessor, CLIPModel
import torch

device = torch.device('cuda')
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model.to(device)


dataset_file_prefix = "dataset/CelebA"
head_class = [200, 200, 500, 500]
tail_count = [3,   5,   3,   5,]


def get_sorted_class_list():
    class_count = []
    for item in os.listdir(dataset_file_prefix + "/CelebA_byclass"):
        class_count.append([item, len(os.listdir(dataset_file_prefix + "/CelebA_byclass/" + item))])

    class_count.sort(key=lambda x: x[1], reverse=True)
    return class_count


def build_dataset(head_class_count, tail_count, class_count, batch_size):
    output_feat = None
    output_label = np.array([], dtype=np.int32)
    file_name_buffer = np.array([])

    for head_id, _ in class_count[0:head_class_count]:
        path = dataset_file_prefix + "/CelebA_byclass/" + str(head_id)
        for pic in os.listdir(path):
            file_name_buffer = np.append(file_name_buffer, path + "/" + pic + "_o")
            file_name_buffer = np.append(file_name_buffer, path + "/" + pic + "_f")
            file_name_buffer = np.append(file_name_buffer, path + "/" + pic + "_bo")
            file_name_buffer = np.append(file_name_buffer, path + "/" + pic + "_bf")

            output_label = np.append(output_label, [int(head_id), int(head_id), int(head_id), int(head_id)]).astype(np.int32)

    for head_id, _ in class_count[head_class_count:-1]:
        path = dataset_file_prefix + "/CelebA_byclass/" + str(head_id)
        pics = os.listdir(path)

        if len(pics) > tail_count:
            pics = np.random.choice(pics, tail_count, replace=False)

        for pic in pics:
            file_name_buffer = np.append(file_name_buffer, path + "/" + pic)
            output_label = np.append(output_label, int(head_id))

    file_name_buffer = np.pad(file_name_buffer, (0, batch_size - (len(file_name_buffer) % batch_size)), mode="constant", constant_values="")
    file_name_buffer = file_name_buffer.reshape((-1, batch_size))

    images = []
    for batch in file_name_buffer:
        for pic in batch:
            if pic == "":
                break
            elif pic.endswith("_o"):
                image = Image.open(pic[:-2])
                images.append(image)
            elif pic.endswith("_f"):
                image = Image.open(pic[:-2])
                fliped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                images.append(fliped_image)
            elif pic.endswith("_bo"):
                image = Image.open(pic[:-3])
                blurred_image = image.filter(ImageFilter.GaussianBlur(radius=3))
                images.append(blurred_image)
            elif pic.endswith("_bf"):
                image = Image.open(pic[:-3])
                fliped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                blurred_flipped_image = fliped_image.filter(ImageFilter.GaussianBlur(radius=3))
                images.append(blurred_flipped_image)
            else:
                image = Image.open(pic)
                images.append(image)

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt")
            inputs.to(device)
            outputs = model.get_image_features(**inputs)

            images.clear()
            last_hidden_states = outputs.cpu()
            output_feat = last_hidden_states.detach().numpy() if output_feat is None else np.concatenate((output_feat, last_hidden_states.detach().numpy()), axis=0)
    np.save(dataset_file_prefix + "/CelebA_" + f"{head_class_count}_{tail_count}" + "/feature", output_feat)
    np.save(dataset_file_prefix + "/CelebA_" + f"{head_class_count}_{tail_count}" + "/label", output_label)


if __name__ == '__main__':
    try:
        for i in range(len(head_class)):
            os.mkdir(dataset_file_prefix + "/CelebA_" + str(head_class[i]) + "_" + str(tail_count[i]))
    except:
        pass
    classes = get_sorted_class_list()
    for i in range(len(head_class)):
        build_dataset(head_class_count=head_class[i], tail_count=tail_count[i], class_count=classes, batch_size=1500)
