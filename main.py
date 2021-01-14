import json
import requests
import urllib.request
import shutil
import os

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

response = requests.get("https://cc0textures.com/api/v1/full_json?category=&date=&q=&method=&type=PhotoTexturePBR"
                        "&sort=Popular", headers=headers)
response = json.loads(response.text)
tags = dict()
idx = 1
l = len(response['Assets'])
for key in response['Assets']:
    tags[key] = response['Assets'][key]['Tags']

    if os.path.exists(f'./textures/{key}/'):
        print(f'{idx}/{l} {key} ok')
        idx += 1
        continue

    print(f'{idx}/{l} {key}')
    # zip url
    if '1K-PNG' in response['Assets'][key]['Downloads']:
        url = response['Assets'][key]['Downloads']['1K-PNG']
    elif '1K-JPG' in response['Assets'][key]['Downloads']:
        url = response['Assets'][key]['Downloads']['1K-JPG']
    else:
        if 'substance' in key.lower():
            print('skipping substance material without exported textures')
            continue
        url = response['Assets'][key]['Downloads'][next(iter(response['Assets'][key]['Downloads']))]

    url = url['RawDownloadLink']
    # download
    zip_file = f'./textures/{key}.zip'
    target_dir = f'./textures/{key}/'

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent',
                          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, zip_file)
    # unzip
    shutil.unpack_archive(zip_file, target_dir)
    idx += 1

# clean-up tags
all_tags = dict()
all_tags_list = []
for key in tags:
    tags[key] = tags[key].split(',')
    tags[key] = filter(lambda x: all(not char.isdigit() for char in x), tags[key])
    tags[key] = list(map(lambda x: x.lower(), tags[key]))
    for tag in tags[key]:
        all_tags_list.append(tag)
        if tag not in all_tags:
            all_tags[tag] = 0
        all_tags[tag] += 1

with open('./tags.json', 'w') as f:
    json.dump(tags, f)

all_tags = dict(sorted(all_tags.items(), key=lambda item: item[1], reverse=True))

print(all_tags)
print(json.dumps(all_tags))

print(len(all_tags), " different labels")
print(len(all_tags_list), " total labels")
print(len(tags), " different textures")

print(list(all_tags.items())[:100])

import matplotlib.pyplot as plt

plt.hist(all_tags_list)
plt.show()
