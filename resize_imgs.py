import os
import nibabel as nib


def crop_nifti(img_path, out_path, target_size):
    img = nib.load(img_path)

    cropped_img = img.slicer[0: target_size[0], 0: target_size[1], 0: target_size[2]]

    nib.save(cropped_img, out_path)


def get_size_stats(input_dir):
    size_stats = {}
    for img_name in os.listdir(input_dir):
        img = nib.load(os.path.join(input_dir, img_name)).get_fdata()
        shape = img.shape
        if shape in size_stats.keys():
            size_stats[shape] += 1
        else:
            size_stats[shape] = 1

    return size_stats


if __name__ == "__main__":
    source_dir = "Data/images"
    target_dir = "Data/images_resized"

    print(get_size_stats(source_dir))

    for filename in os.listdir(source_dir):
        crop_nifti(os.path.join(source_dir, filename), os.path.join(target_dir, filename), (512, 307, 1))
    print(get_size_stats(target_dir))
