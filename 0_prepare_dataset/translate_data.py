import pandas as pd
import warnings
from tqdm import tqdm
import json
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration

warnings.filterwarnings("ignore")
df = pd.read_json("../data/chat.json")
df["translated_conversations"] = df["conversations"]
device = "cuda:0"
batch_size = 32


def process_batches(df: pd.DataFrame, batch_size=batch_size) -> list:
    """Batches list of data for training, converting each batch into a list of dictionaries."""
    batches_list = []
    num_batches = df.shape[0] // batch_size
    for i in tqdm(range(num_batches)):
        ldx = i * batch_size
        batch_dict = df.iloc[ldx : ldx + batch_size].to_dict("records")
        batches_list.append(batch_dict)

    # Check if there are remaining rows to form a batch and process them
    remaining_rows = df.shape[0] % batch_size
    if remaining_rows > 0:
        last_batch_dict = df.iloc[-remaining_rows:].to_dict("records")
        batches_list.append(last_batch_dict)

    return batches_list


def merge_content(user_content, assistant_content, batch_size=batch_size):
    merged_chat = []

    for i in range(batch_size):
        user_message = [
            {"role": "user", "content": user_content[i]},
            {"role": "assistant", "content": assistant_content[i]},
        ]
        merged_chat.append(user_message)

    return merged_chat


def clean_batches(batches, trans_token="<2pl>"):
    # Define a function to clean each dictionary
    def clean_dict(item):
        cleaned_translated_conversations = [
            {
                "value": trans_token
                + conv["value"].replace("<image>", "").replace("\n", "")
            }
            for conv in item["translated_conversations"]
        ]
        item["translated_conversations"] = cleaned_translated_conversations
        return item
        # return {"translated_conversations": cleaned_translated_conversations}

    cleaned_batches = []
    for batch in tqdm(batches):
        # Assuming 'batch' is a list of dictionaries as described in your example
        cleaned_batch = list(map(clean_dict, batch))
        cleaned_batches.append(cleaned_batch)

    return cleaned_batches


# Note: The provided example assumes that 'batches' is a list of lists (where each inner list represents a batch),
# and each element in an inner list is a dictionary with a key 'translated_conversations' pointing to a list of
def randomly_append_image(text):
    choices = ["<image>\n", "\n<image>"]
    position = random.choice(["start", "end"])
    image_str = random.choice(choices)

    if position == "start":
        return image_str + text
    else:
        return text + image_str


model_name = "jbochi/madlad400-3b-mt"
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map=device)
tokenizer = T5Tokenizer.from_pretrained(model_name, device_map=device)

batches = process_batches(df=df, batch_size=batch_size)
cleaned_batches = clean_batches(batches)
with open("../data/translated_data_2.jsonl", "w") as f:
    for b in tqdm(cleaned_batches):
        try:
            encoded_human = tokenizer(
                [text["translated_conversations"][0]["value"] for text in b],
                return_tensors="pt",
                max_length=64,
                padding=True,
            )
            encoded_ai = tokenizer(
                [text["translated_conversations"][1]["value"] for text in b],
                return_tensors="pt",
                max_length=64,
                padding=True,
            )
            encoded_text = encoded_human.to("cuda")
            human_translations = tokenizer.batch_decode(
                model.generate(**encoded_text, max_new_tokens=96),
                skip_special_tokens=True,
            )
            encoded_text = encoded_ai.to("cuda")
            assistant_translations = tokenizer.batch_decode(
                model.generate(**encoded_text, max_new_tokens=96),
                skip_special_tokens=True,
            )
            human_translations = [
                randomly_append_image(at) for at in human_translations
            ]
            chat = merge_content(human_translations, assistant_translations)
            print(chat)
            for i, instruction in enumerate(chat):
                batch_copy = b
                batch_copy[i]["translated_conversations"] = instruction
                json.dump(batch_copy[i], f)
                f.write("\n")
        except:
            "ERROR"
