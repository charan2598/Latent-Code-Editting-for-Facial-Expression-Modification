import os
import json
import shutil
from tqdm import tqdm
import cv2

json_dir = "..\\ffhq-features-dataset\\json"
images_dir = "..\\ffhq-dataset\\thumbnails128x128"
out_dir = ".\\data"

output_resolution = 512

json_files = os.listdir(json_dir)
images_sub_dir = os.listdir(images_dir)

anger = os.path.join(out_dir, 'anger')
os.makedirs(anger, exist_ok=True)
contempt = os.path.join(out_dir, 'contempt')
os.makedirs(contempt, exist_ok=True)
disgust = os.path.join(out_dir, 'disgust')
os.makedirs(disgust, exist_ok=True)
fear = os.path.join(out_dir, 'fear')
os.makedirs(fear, exist_ok=True)
happiness = os.path.join(out_dir, 'happiness')
os.makedirs(happiness, exist_ok=True)
neutral = os.path.join(out_dir, 'neutral')
os.makedirs(neutral, exist_ok=True)
sadness = os.path.join(out_dir, 'sadness')
os.makedirs(sadness, exist_ok=True)
surprise = os.path.join(out_dir, 'surprise')
os.makedirs(surprise, exist_ok=True)

map_dir = {"anger": anger, "contempt":contempt, "disgust":disgust, "fear":fear, "happiness":happiness, "neutral":neutral, "sadness":sadness, "surprise":surprise}

def max_emotion(d):
    max_key = ""
    max_score = -1.0
    for key in d.keys():
        if d[key] > max_score:
            max_score = d[key]
            max_key = key
    return max_key, max_score

for json_file in tqdm(json_files):
    filename = json_file[:-5]
    filename_int = int(filename)
    image_file_dir = "%05d"%(int(filename_int//1000)*1000)
    with open(os.path.join(json_dir, json_file), 'r') as file_hand:
        file_contents = file_hand.read()
        json_content = json.loads(file_contents)
        if len(json_content) > 0:
            emotion, emotion_score = max_emotion(json_content[0]["faceAttributes"]["emotion"])
            source = os.path.join(images_dir, image_file_dir, filename+".png")
            destination = os.path.join(map_dir[emotion], filename+".png")
            # shutil.copyfile(os.path.join(images_dir, image_file_dir, filename+".png"), os.path.join(map_dir[emotion], filename+".png"))
            image = cv2.imread(source, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (output_resolution, output_resolution), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(destination, image)
