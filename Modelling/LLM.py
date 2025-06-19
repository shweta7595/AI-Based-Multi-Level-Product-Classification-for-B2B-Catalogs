#!/usr/bin/env python
# coding: utf-8

# Convert to jsonl file

# In[ ]:


import pandas as pd
import json
from google.colab import files

# 读取 Excel 文件（也支持 CSV，替换成 pd.read_csv 即可）
file_path = "/content/Test_Set.xlsx"
df = pd.read_excel(file_path)

# 拼接三段产品描述字段
def build_input(row):
    d1 = str(row.get("product_desc_1", "")).strip()
    d2 = str(row.get("product_desc_2", "")).strip()
    d3 = str(row.get("central_description", "")).strip()
    return f"{d1} {d2} {d3}".strip()

df["input_text"] = df.apply(build_input, axis=1)

# 删除分类或描述缺失的行
df = df.dropna(subset=["FTICategory1", "FTICategory2", "FTICategory3", "input_text"])

# 可选：最多5000条
df = df.iloc[:5000]

# 输出文件路径，文件名为 1.jsonl
output_path = "/content/1.jsonl"

# 写入符合 deepseek_chat 格式的 jsonl 文件
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please classify the following product description into three category levels: {row['input_text']}"
                },
                {
                    "role": "assistant",
                    "content": f"Level 1: {row['FTICategory1']}\nLevel 2: {row['FTICategory2']}\nLevel 3: {row['FTICategory3']}"
                }
            ]
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Done! File saved as: 1.jsonl")

# 自动触发下载
files.download(output_path)



# In[ ]:





# In[ ]:


# 指定两个输入文件路径
input_file_1 = "/content/1.jsonl"
input_file_2 = "/content/deepseek_chattest_format.jsonl"

# 输出文件路径
output_file = "/content/training_final.jsonl"

# 打开输出文件进行写入
with open(output_file, "w", encoding="utf-8") as outfile:
    for file_path in [input_file_1, input_file_2]:
        with open(file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                outfile.write(line)

print("✅ 合并完成：文件保存为 training_final.jsonl")

# 自动下载合并后的文件
from google.colab import files
files.download(output_file)


# In[ ]:


import json
from google.colab import files

# 原始文件路径（你上传的）
input_path = "/content/training_final.jsonl"

# 输出文件路径
output_path = "/content/training_final_updated.jsonl"

# 替换后的新提示语
new_prompt_prefix = "Please extract the Level 1, Level 2, and Level 3 categories from the product description below: "

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        obj = json.loads(line)

        # 修改 user 的 content 字段（假设在 messages[0] 中）
        if obj.get("messages") and obj["messages"][0]["role"] == "user":
            original = obj["messages"][0]["content"]
            # 提取原始 product description（提示语后的部分）
            if ":" in original:
                desc = original.split(":", 1)[1].strip()
            else:
                desc = original.strip()
            obj["messages"][0]["content"] = new_prompt_prefix + desc

        outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ Prompt 替换完成，文件已保存为：training_final_updated.jsonl")

files.download(output_path)


# In[ ]:


get_ipython().system('pip install --upgrade openai')


# LLAMA_FACTORY
# 

# In[ ]:


# 🚀 Clone LLaMA-Factory and install dependencies
get_ipython().system('git clone https://github.com/hiyouga/LLaMA-Factory.git')
get_ipython().run_line_magic('cd', 'LLaMA-Factory')
get_ipython().system('pip install -r requirements.txt')

# 📁 Move your uploaded dataset (e.g., 1.jsonl) into the dataset folder
import shutil
shutil.move("/content/1.jsonl", "./data/qwen_chat/1.jsonl")  # Adjust if your filename is different

# 🧠 Run fine-tuning with QLoRA on Qwen2.5-14B-Instruct
get_ipython().system('python src/train_bash.py    --stage sft \\                                # Stage: Supervised Fine-tuning (SFT)')
  --model_name_or_path Qwen/Qwen2.5-14B-Instruct \  # Pretrained base model from HuggingFace
  --do_train \                                 # Enable training
  --dataset_dir ./data \                       # Directory where your dataset is stored
  --dataset qwen_chat \                        # Dataset name (folder under dataset_dir)
  --template qwen \                            # Chat template (ChatML style for Qwen)
  --finetuning_type lora \                     # Use LoRA for parameter-efficient tuning
  --lora_target all \                          # Apply LoRA to all tunable layers
  --output_dir ./output_qwen14b \              # Output directory for saving model checkpoints
  --cutoff_len 2048 \                          # Max token length for input sequences
  --learning_rate 0.00005 \                    # Learning rate (safe and effective for SFT)
  --num_train_epochs 3 \                       # Number of training epochs
  --per_device_train_batch_size 1 \            # Batch size per GPU (keep small to avoid OOM)
  --gradient_accumulation_steps 16 \           # Accumulate gradients to simulate larger batch size
  --lr_scheduler_type cosine \                 # Use cosine learning rate scheduler
  --logging_steps 10 \                         # Log training loss every 10 steps
  --save_steps 100 \                           # Save checkpoints every 100 steps
  --warmup_ratio 0.05 \                        # Warm-up ratio for learning rate scheduler
  --bf16 \                                     # Use bfloat16 for faster training and lower memory
  --plot_loss                                  # Save a plot of the training loss curve


# Huggingface&LORA

# In[ ]:


# Step 1: Install dependencies
get_ipython().system('pip install -q transformers peft accelerate datasets')

# Step 2: Import packages
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Step 3: Load Qwen tokenizer and base model
model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Step 4: Apply LoRA configuration to the model
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Show trainable parameters for verification

# Step 5: Define ChatML-style dataset (expects .jsonl file with 'messages' key)
class ChatMLDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                messages = data.get("messages", [])
                if not messages: continue

                conversation = ""
                for msg in messages:
                    role, content = msg.get("role", ""), msg.get("content", "")
                    if role and content:
                        conversation += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                conversation += "<|endoftext|>"

                encoding = tokenizer(conversation, return_offsets_mapping=True, truncation=True, max_length=self.max_length)
                input_ids, offsets = encoding["input_ids"], encoding["offset_mapping"]
                labels = [-100] * len(input_ids)

                for msg in messages:
                    if msg.get("role") == "assistant" and msg.get("content"):
                        start = conversation.find(msg["content"])
                        end = start + len(msg["content"])
                        for i, (s, e) in enumerate(offsets):
                            if s >= start and e <= end and s < e:
                                labels[i] = input_ids[i]
                self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }

# 🔁 Load your dataset here (upload your JSONL to /content/ first)
dataset_path = "/content/1.jsonl"
train_dataset = ChatMLDataset(dataset_path, tokenizer)

# Step 6: Define collate function with padding
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Step 7: Configure training arguments
training_args = TrainingArguments(
    output_dir="qwen14b-lora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch size = 16
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    learning_rate=0.0002,
    bf16=True,
    optim="adamw_torch",
    warmup_steps=100,
    save_total_limit=2,
    report_to="none"
)

# Step 8: Start training with Hugging Face Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn
)
trainer.train()

# Step 9: Save the LoRA adapter only (not full 14B model)
model.save_pretrained("qwen14b-lora-adapter")
print("✅ LoRA fine-tuned adapter saved to: qwen14b-lora-adapter")


# Call & evaluation

# In[ ]:


from openai import OpenAI


client = OpenAI(
    api_key="sk-jgcqexytibwyxpwlpyonvpcclzfnltxupwdyrcypfjcahcjw",
    base_url="https://api.siliconflow.cn/v1"
)

model_id = "ft:LoRA/Qwen/Qwen2.5-14B-Instruct:yn4rxfw1pv:trail:lzxdqroqucnsgosadivv-ckpt_step_738"


# ✅ Product description to be classified (replace with any input text)
product_description = "Universal locking gas cap to prevent fuel theft and/or tampering. Heavy-duty locking gas cap with two keys, designed to fit most vehicles."

# ✅ Prompt the model to classify the description into three category levels
messages = [
    {
        "role": "user",
        "content": f"Please classify the following product description into three category levels (Level1, Level2, Level3): {product_description}"
    }
]

# ✅ Call the model with the classification request
response = client.chat.completions.create(
    model=model_id,
    messages=messages,
    max_tokens=512,
    temperature=0.7
)

# ✅ Print the model's classification result
print("🔍 Model Response:\n")
print(response.choices[0].message.content)



# In[ ]:


# Install required libraries (if not already installed)
get_ipython().system('pip install openai pandas scikit-learn openpyxl')

import pandas as pd
from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# 1. Initialize your OpenAI client

from openai import OpenAI
client = OpenAI(
    api_key="sk-jgcqexytibwyxpwlpyonvpcclzfnltxupwdyrcypfjcahcjw",
    base_url="https://api.siliconflow.cn/v1"
)

model_id = "ft:LoRA/Qwen/Qwen2.5-14B-Instruct:yn4rxfw1pv:ingram_classification:dibujfbkcbgsrypulhwl"

# 2. Load your uploaded test file
file_path = "/content/Test_Set.xlsx"
df = pd.read_excel(file_path)

# 3. Combine description fields into one input text
df["input_text"] = df["product_desc_1"].fillna("") + " " + df["product_desc_2"].fillna("") + " " + df["central_description"].fillna("")

# 4. Define function to call the model for prediction
def predict_category(text):
    prompt = f"Please classify the following product description into three category levels (Level1, Level2, Level3): {text}"
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

# 5. Run predictions (limit to first 100 samples for safety)
df = df.head(200)
df["raw_prediction"] = [predict_category(txt) for txt in tqdm(df["input_text"])]

# 6. Extract Level 1, 2, and 3 predictions from the raw model output
def extract_levels(text):
    if "ERROR" in text:
        return "", "", ""
    level1 = level2 = level3 = ""
    lines = text.split("\n")
    for line in lines:
        if "Level 1" in line or "Level1" in line:
            level1 = line.split(":")[-1].strip()
        elif "Level 2" in line or "Level2" in line:
            level2 = line.split(":")[-1].strip()
        elif "Level 3" in line or "Level3" in line:
            level3 = line.split(":")[-1].strip()
    return level1, level2, level3

df[["pred_level1", "pred_level2", "pred_level3"]] = df["raw_prediction"].apply(lambda x: pd.Series(extract_levels(x)))

# 7. Print accuracy and full classification report for each category level
print("\n📊 Level 1 Accuracy:", accuracy_score(df["FTICategory1"], df["pred_level1"]))
print(classification_report(df["FTICategory1"], df["pred_level1"], zero_division=0))

print("\n📊 Level 2 Accuracy:", accuracy_score(df["FTICategory2"], df["pred_level2"]))
print(classification_report(df["FTICategory2"], df["pred_level2"], zero_division=0))

print("\n📊 Level 3 Accuracy:", accuracy_score(df["FTICategory3"], df["pred_level3"]))
print(classification_report(df["FTICategory3"], df["pred_level3"], zero_division=0))

# 8. Save the full prediction results to a CSV file
df.to_csv("/content/classification_results_with_eval.csv", index=False)
print("\n✅ Saved as classification_results_with_eval.csv")


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

accuracies = {
    "Level 1": accuracy_score(df["FTICategory1"], df["pred_level1"]),
    "Level 2": accuracy_score(df["FTICategory2"], df["pred_level2"]),
    "Level 3": accuracy_score(df["FTICategory3"], df["pred_level3"])
}
levels = list(accuracies.keys())
values = [round(acc * 100, 2) for acc in accuracies.values()]

sns.barplot(x=levels, y=values, palette="Blues_d")
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("📊 Prediction Accuracy per Category Level")
for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.metrics import f1_score

# Compute F1-scores for each level
f1_level1 = f1_score(df["FTICategory1"], df["pred_level1"], average='macro', zero_division=0)
f1_level2 = f1_score(df["FTICategory2"], df["pred_level2"], average='macro', zero_division=0)
f1_level3 = f1_score(df["FTICategory3"], df["pred_level3"], average='macro', zero_division=0)

# Print F1 results
print("📊 Macro F1-scores by level:")
print(f"Level 1 F1-score: {f1_level1:.4f}")
print(f"Level 2 F1-score: {f1_level2:.4f}")
print(f"Level 3 F1-score: {f1_level3:.4f}")


# In[ ]:


print("📊 Label distribution:")
print(df["FTICategory3"].value_counts())


# In[ ]:


from sklearn.metrics import confusion_matrix

plt.figure(figsize=(10, 8))
cm = confusion_matrix(df["FTICategory1"], df["pred_level1"], labels=df["FTICategory1"].unique())
sns.heatmap(cm, annot=False, fmt='d', cmap="YlGnBu", xticklabels=True, yticklabels=True)
plt.title("🧯 Confusion Matrix for Level 1")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.metrics import f1_score

# Compute macro, micro, and weighted F1-scores for all 3 levels
f1_results = {}

for level in [1, 2, 3]:
    y_true = df[f"FTICategory{level}"]
    y_pred = df[f"pred_level{level}"]

    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    f1_results[f"Level {level}"] = {
        "macro": f1_macro,
        "micro": f1_micro,
        "weighted": f1_weighted
    }

# Print the results
print("📊 F1-scores by category level:")
for level, scores in f1_results.items():
    print(f"\n{level}:")
    print(f"  Macro F1-score:    {scores['macro']:.4f}")
    print(f"  Micro F1-score:    {scores['micro']:.4f}")
    print(f"  Weighted F1-score: {scores['weighted']:.4f}")


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, zero_division=0))


# In[ ]:


df_errors = df[df["FTICategory3"] != df["pred_level3"]]

error_counts = (
    df_errors.groupby(["FTICategory3", "pred_level3"])
    .size()
    .reset_index(name="count")
    .sort_values(by="count", ascending=False)
)

print("📉 Top 10 Most Frequent Misclassifications (Level 3):")
print(error_counts.head(10))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("/content/classification_results_with_eval.csv")

def get_top_errors(df, true_col, pred_col, level_name, top_n=5):
    errors = df[df[true_col] != df[pred_col]].copy()
    errors["True Label"] = errors[true_col].astype(str)
    errors["Predicted Label"] = errors[pred_col].astype(str)
    errors["Label Pair"] = errors["True Label"] + " → " + errors["Predicted Label"]
    counts = (
        errors.groupby("Label Pair")
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )
    counts["Level"] = level_name
    return counts.head(top_n)

top1 = get_top_errors(df, "FTICategory1", "pred_level1", "Level 1")
top2 = get_top_errors(df, "FTICategory2", "pred_level2", "Level 2")
top3 = get_top_errors(df, "FTICategory3", "pred_level3", "Level 3")

level_colors = {
    "Level 1": "#1f77b4",
    "Level 2": "#2ca02c",
    "Level 3": "#d62728",
}

for top_errors in [top1, top2, top3]:
    level = top_errors["Level"].iloc[0]
    color = level_colors[level]

    top_errors = top_errors.sort_values(by="count", ascending=True)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_errors["Label Pair"], top_errors["count"], color=color)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}', va='center', fontsize=9)

    plt.title(f"Top 5 Misclassifications — {level}", fontsize=14)
    plt.xlabel("Misclassification Count")
    plt.ylabel("True → Predicted Label")
    plt.tight_layout()
    plt.show()

