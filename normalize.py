import numpy as np
import os
import json
from autocorrect import Speller
from random import randrange
from PIL import Image

# target image size in pixels
IMAGE_SIZE = 64

# whether to generate 90 degrees rotated images to dataset
AUGMENTATION_ROTATE = False

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
IMAGE_MODE = 'crop'

# number of crops from one texture
CROP_COUNT = 10

# minimum size of crop before downscaling to target image size
CROP_SIZE_MIN = 64

# maximum crop size, set to same values as `CROP_SIZE_MIN` if you want specific size
CROP_SIZE_MAX = 256

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
KEEP_TAGS = {
    "stone", "wood", "metal"
}


def normalize_image(im_f):
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
        if AUGMENTATION_ROTATE:
            rotate90 = pil_img.rotate(90)
            rotate90 = np.asarray(rotate90)
            rotate90 = rotate90.reshape((IMAGE_SIZE, IMAGE_SIZE, 3))
            rotate90 = rotate90 / 255.0
            rotate90 = np.float16(rotate90)
            result.append(rotate90)

        npa = np.asarray(pil_img)
        npa = npa.reshape((IMAGE_SIZE, IMAGE_SIZE, 3))
        npa = npa / 255.0
        npa = np.float16(npa)
        result.append(npa)

    image = Image.open(im_f)  # type: Image.Image

    if SKIP_TRANSPARENT:
        if has_transparency(image):
            return []

    # convert to rgb
    image = image.convert('RGB')

    if IMAGE_MODE == 'downscale':
        # downscale to common resolution
        resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=RESIZE_FILTER)

        # write input data in some good format ready for training
        add(resized)
    elif IMAGE_MODE == 'crop':
        for _ in range(CROP_COUNT):
            random_crop_size = CROP_SIZE_MIN if CROP_SIZE_MIN == CROP_SIZE_MAX else np.random.randint(CROP_SIZE_MIN,
                                                                                                      CROP_SIZE_MAX)
            crop_size = np.min([image.width, image.height, random_crop_size])
            if image.width < random_crop_size or image.height < random_crop_size:
                print(f'Image {file.name} {image.width}x{image.height} is smaller'
                      f' than crop size {random_crop_size}, it may be upscaled!')

            # generate random crop start point
            x1 = 0 if image.width == crop_size else randrange(0, image.width - crop_size)
            y1 = 0 if image.height == crop_size else randrange(0, image.height - crop_size)

            crop = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            resized = crop.resize((IMAGE_SIZE, IMAGE_SIZE), resample=RESIZE_FILTER)

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

for file in os.scandir('./textures'):
    if os.path.isdir(file):
        for entry in os.scandir(f'./textures/{file.name}/'):
            if any(x in entry.name.lower() for x in
                   ['_col.', '_color.', 'diffuse.', '_albedo.', '_basecolor.']) and entry.is_file():

                print(len(arr))

                results = normalize_image(f'./textures/{file.name}/{entry.name}')

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
