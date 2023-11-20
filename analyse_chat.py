import sys
from glob import glob
from bs4 import BeautifulSoup
from tqdm import tqdm
from utils import collect_profile, create_profile_image
import os


exported_chat = sys.argv[1]
list_of_htmls = glob(f"{exported_chat}/messag*")

print("Extracting messages...")
users_and_messages = {}
for html in tqdm(list_of_htmls):
    print(list_of_htmls)
    with open(html, "r") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content)
    all_messages = soup.findAll("div", {"class": "body"})
    users_messages = [
        tag for tag in all_messages if tag.findAll("div", {"class": "from_name"})
    ]
    for message in users_messages:
        name_tag = message.find("div", {"class": "from_name"})
        username = name_tag.contents[0].split("\n")[1]
        if username not in users_and_messages.keys():
            users_and_messages[username] = []
        text_tag = message.find("div", {"class": "text"})
        if text_tag:
            user_message = text_tag.contents[0].split("\n")[1]
            users_and_messages[username].append(user_message)


os.makedirs(exported_chat + "neural_analys_result/", exist_ok=True)

print("Analysing messages...")
for username, user_messages in tqdm(users_and_messages.items()):
    if len(user_messages):
        profile = collect_profile(user_messages)
        save_path = (
            exported_chat + "neural_analys_result/" + username.replace("/", "") + ".png"
        )
        create_profile_image(profile, save_path)

print("Done! Analytics is in folder", exported_chat + "neural_analys_result/")
