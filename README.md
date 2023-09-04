# Telegram chat analyser

This app analyse telegram chats in RUSSIAN LANGUAGE!

The program analyzes the exported chats from telegram, creating a folder with the profiles of each user in the folder with the exported chats.

Profiles are compiled using neural networks that analyze messages.

It turns out something like this:

![Alt text](example.png?raw=true "Title")

**Explanations:**
1. Starting with neutrality and ending with anger - this is the emotional coloring of messages
2. Inappropriate statements - the percentage of such statements that, for example, justify murder or offend the feelings of believers
3. Negative attitude measures the tone of messages and shows how negative a person is in the chat
4. Toxic messages -- those that make other people feel worse


## Launch

```commandline
conda env create -f environment.yml
conda activate tg_chat_analyser
python analyse_chat.py ./REL_PATH/TO_EXPORTED/CHAT_FOLDER/
```
*please don't forget to add slash (/) in the end*

After launching, the neural_analys_result folder will appear in the chat folder -- the analysis result will be there