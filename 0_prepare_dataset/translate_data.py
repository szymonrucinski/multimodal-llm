import pandas as pd
import warnings
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import json

warnings.filterwarnings("ignore")

# Convert JSON to DataFrame
df = pd.read_json("../data/chat.json")

df["conversations"]


def process_batches(df: pd.DataFrame, batch_size=32) -> list:
    """Batches list of data for training"""
    batches_list = []
    for i in range(df.shape[0] // batch_size):
        ldx = i * batch_size
        batches_list.append(df["conversations"][ldx : ldx + batch_size])

    batches_list.append(df[-(df.shape[0] % batch_size) :])
    return batches_list


def merge_content(user_content, assistant_content):
    batch_size = 32
    merged_chat = []

    for i in range(batch_size):
        user_message = [
            {"role": "user", "content": user_content[i]},
            {"role": "assistant", "content": assistant_content[i]},
        ]
        merged_chat.append(user_message)

    return merged_chat


model_name = "jbochi/madlad400-3b-mt"
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="cuda:0")
tokenizer = T5Tokenizer.from_pretrained(model_name, device_map="cuda:0")

batches = process_batches(df=df, batch_size=32)
trans_token = "<2pl> "
with open("../data/translated_data.jsonl", "w") as f:
    for b in tqdm(batches):
        encoded_human = tokenizer(
            b.apply(
                lambda x: trans_token
                + x[0]["value"].replace("<image>", "").replace("\n", "")
            ).to_list(),
            return_tensors="pt",
            max_length=64,
            padding=True,
        )
        encoded_ai = tokenizer(
            b.apply(lambda x: trans_token + x[1]["value"])
            .replace("<image>", "")
            .replace("\n", "")
            .to_list(),
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

        chat = merge_content(human_translations, assistant_translations)
        for item in chat:
            json.dump(item, f)
            f.write("\n")
