import numpy as np
import os
import json
from autocorrect import Speller
from random import randrange
from PIL import Image

# target image size in pixels
IMAGE_SIZE = 64

# interpolation filter used to downscale the images
RESIZE_FILTER = Image.BICUBIC

# whether to auto-spell-correct tags
TAG_AUTOCORRECT = True

# how many of mostly used tags to output
TOP_N_TAGS = 10

# whether to skip transparent objects
SKIP_TRANSPARENT = True

# skip images that have 0 tags
SKIP_IMAGES_WITH_NO_TAGS = True

# mode of creation of images: downscale, crop
IMAGE_MODE = 'downscale'

# number of crops from one texture
CROP_COUNT = 5

# minimum size of crop before downscaling to target image size
CROP_SIZE_MIN = 128

# maximum crop size, set to same values as `CROP_SIZE_MIN` if you want specific size
CROP_SIZE_MAX = 128

# list of groups of tags that should be merged to one tag
TAG_UNIFICATION = {
    'wood': {'wood', 'wooden'},
    'paving': {'paving', 'pavement', 'paved'},
    'metal': {'metal', 'steel'},
    'tiles': {'tiled', 'tiles'},
    'smooth': {'fine', 'smooth', 'clean'},
    'stone': {'stone', 'stones'},
    'paint': {'paint', 'painted'},
    'floor': {'floor', 'flooring'},
    'cloth': {'cloth', 'fabric', 'clothing'},
    'rust': {'rusty', 'rust'},
    'brick': {'brick', 'bricks'},
    'crack': {'crack', 'cracks', 'cracked'},
    'road': {'road', 'street'},
    'planks': {'planks', 'herringbone', 'parque'},
    'gravel': {'gravel', 'pebbles'},
    'dirt': {'dirt', 'dirty'},
    'danger': {'danger', 'warning'},
    'scratches': {'scratched', 'scratches'},
    'gray': {'gray', 'grey'},
}

# tags to skip and never output
SKIP_TAGS = {'shiny'}

# tags to keep
KEEP_TAGS = {}


def normalize_image(im_al, im_nr):
    result = []

    def has_transparency(img):
        if img.mode == "P":
            transparent = img.info.get("transparency", -1)
            for _, index in img.getcolors():
                if index == transparent:
                    return True
        elif img.mode == "RGBA":
            extrema = img.getextrema()
            if extrema[3][0] < 255:
                return True

        return False

    def add(pil_img):
        npa = np.asarray(pil_img)
        npa = npa.reshape((IMAGE_SIZE, IMAGE_SIZE, 6))
        npa = npa / 255.0
        npa = np.float16(npa)
        result.append(npa)

    image_al = Image.open(im_al)  # type: Image.Image
    image_nr = Image.open(im_nr)  # type: Image.Image

    if SKIP_TRANSPARENT:
        if has_transparency(image_al):
            return []

    # convert to rgb
    image_al = image_al.convert('RGB')
    image_nr = image_nr.convert('RGB')

    if IMAGE_MODE == 'downscale':
        # downscale to common resolution
        resized_al = image_al.resize((IMAGE_SIZE, IMAGE_SIZE), resample=RESIZE_FILTER)
        resized_nr = image_nr.resize((IMAGE_SIZE, IMAGE_SIZE), resample=RESIZE_FILTER)

        resized = np.dstack([resized_al, resized_nr])

        # write input data in some good format ready for training
        add(resized)
    elif IMAGE_MODE == 'crop':
        for _ in range(CROP_COUNT):
            random_crop_size = CROP_SIZE_MIN if CROP_SIZE_MIN == CROP_SIZE_MAX else np.random.randint(CROP_SIZE_MIN,
                                                                                                      CROP_SIZE_MAX)
            crop_size = np.min([image_al.width, image_al.height, random_crop_size])
            if image_al.width < random_crop_size or image_al.height < random_crop_size:
                print(f'Image {file.name} {image_al.width}x{image_al.height} is smaller'
                      f' than crop size {random_crop_size}, it may be upscaled!')

            # generate random crop start point
            x1 = 0 if image_al.width == crop_size else randrange(0, image_al.width - crop_size)
            y1 = 0 if image_al.height == crop_size else randrange(0, image_al.height - crop_size)

            crop_al = image_al.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            crop_nr = image_nr.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            resized_al = crop_al.resize((IMAGE_SIZE, IMAGE_SIZE), resample=RESIZE_FILTER)
            resized_nr = crop_nr.resize((IMAGE_SIZE, IMAGE_SIZE), resample=RESIZE_FILTER)

            resized = np.dstack([resized_al, resized_nr])

            add(resized)

    return result


spell = Speller()

# initialize np array
arr = []
arr_labels = []

# load tags
all_tags = dict()
all_tags_count = dict()
total_tags = 0
with open('./tags.json') as f:
    tags = json.load(f)

# open color textures
tag_unif_inv = {vi: k for k, v in TAG_UNIFICATION.items() for vi in v}

print(tag_unif_inv)

ALBEDO_STRINGS = ['_col.', '_color.', 'diffuse.', '_albedo.', '_basecolor.', 'col.']
NORMAL_STRINGS = ['_nrm.', '_normal.', '_normalmap.', 'norm.']

for file in os.scandir('./textures'):
    if os.path.isdir(file):
        albedo = None
        normal = None

        for entry in os.scandir(f'./textures/{file.name}/'):
            if any(x in entry.name.lower() for x in NORMAL_STRINGS) and entry.is_file():
                normal = entry.name.lower()
            elif any(x in entry.name.lower() for x in ALBEDO_STRINGS) and entry.is_file():
                albedo = entry.name.lower()

        print(len(arr))

        if albedo is None or normal is None:
            continue

        results = normalize_image(f'./textures/{file.name}/{albedo}', f'./textures/{file.name}/{normal}')

        # skip if we did not output any images
        if len(results) == 0:
            continue

        # normalize tag data
        img_tags = []
        for tag in tags[file.name]:
            tag = tag.strip().lower()

            if TAG_AUTOCORRECT:
                tag = spell(tag)

            # if we have unification for this tag, apply it
            if tag in tag_unif_inv:
                tag = tag_unif_inv[tag]

            if tag in SKIP_TAGS:
                continue

            if len(KEEP_TAGS) > 0 and tag not in KEEP_TAGS:
                continue

            if tag not in all_tags:
                all_tags[tag] = tag
                all_tags_count[tag] = 0
            img_tags.append(all_tags[tag])

        # skip images with no tags
        if SKIP_IMAGES_WITH_NO_TAGS and len(img_tags) == 0:
            continue

        for result in results:
            arr.append(result)

        # output tags for every training (augmented) image created from this source image
        for _ in range(len(results)):
            uniq_tags = list(set(img_tags))

            for tag in uniq_tags:
                all_tags_count[tag] += 1
                total_tags += 1

            arr_labels.append(uniq_tags)

print(len(all_tags_count), "different tags")
print(total_tags, "total tags")
all_tags_count = dict(sorted(all_tags_count.items(), key=lambda item: item[1], reverse=True))

with open('./top_tags.json', 'w') as f:
    json.dump(all_tags_count, f, indent=2)

for not_top_tag in list(all_tags_count.keys())[TOP_N_TAGS:]:
    for example_tags in arr_labels:
        if not_top_tag in example_tags:
            total_tags -= 1
            example_tags.remove(not_top_tag)

print(TOP_N_TAGS, "different tags (after top n)")
print(total_tags, "total tags (after top n)")

np_arr = np.array(arr)
print('X', np_arr.shape)

np_arr_labels = np.array(arr_labels)
print('Y', np_arr_labels.shape)

np.save(f'./imgs_{IMAGE_SIZE}_{RESIZE_FILTER}.npy', np_arr)
np.save(f'./labels_{IMAGE_SIZE}_{RESIZE_FILTER}.npy', np_arr_labels)

print(json.dumps(list(all_tags_count.keys())[:TOP_N_TAGS]))
