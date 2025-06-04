import os
import pandas as pd
from openai import OpenAI

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
# Base path where your CSV and output files reside
BASE_PATH = r"C:\Users\serge\Downloads\tmp"

# Define the dimensions we want to score and their descriptions
parameters = {
    "Negativity": "Degree of negative emotional tone expressed in the text",
    "Controversiality": "Extent to which the text is likely to provoke disagreement or strong opposing views",
    "Emotional Elevation/Excitement": "Level of emotional intensity or stimulation conveyed by the text",
    "Overall Quality": "Clarity, coherence, and polish of the text's expression"
}

# Join parameter names into a comma-separated string for the prompt
parameters_text = ", ".join(parameters.keys())

# Construct the system/user prompt template. We will append each transcript to this.
prompt_template = rf"""Analyze the following text and assign a score from 1 to 10 for each of the following dimensions:
    {parameters}
Respond with four numbers only, in the following order:
    {parameters_text}
Separate each number with a comma. Do not include any explanation or extra text.

"""

# ─── LOAD YOUR DATA ────────────────────────────────────────────────────────────
# Read the CSV that contains the transcripts (and any other columns you have)
input_csv_path = os.path.join(BASE_PATH, "youtube_with_sentiment.csv")
df = pd.read_csv(input_csv_path)

# ─── INITIALIZE OPENAI CLIENT ───────────────────────────────────────────────────
# Make sure your environment variable OPENAI_API_KEY is set, or pass it explicitly.
client = OpenAI()

# ─── PREPARE A DATAFRAME TO HOLD THE SCORES ────────────────────────────────────
# We'll store the four numeric scores for each dimension in this new DataFrame
new_data = pd.DataFrame(columns=parameters.keys())

# ─── LOOP OVER EACH ROW, QUERY GPT, AND PARSE THE RESPONSE ────────────────────
for idx, row in df.iterrows():
    # 1) Extract the transcript and remove any stray double quotes
    transcript_text = row["transcript"].replace('"', "")

    # 2) Build the full prompt by appending the transcript to our template
    full_prompt = prompt_template + transcript_text

    # 3) Send the request to the ChatGPT model
    response = client.responses.create(
        model="gpt-4.1",
        input=full_prompt
    )

    # 4) The model’s raw output: e.g., "2, 7, 5, 8"
    resp_text = response.output_text.strip()
    print(f"{idx}:\t{resp_text}")

    # 5) Split the comma-separated numbers and assign to the new_data DataFrame
    scores = [s.strip() for s in resp_text.split(",")]
    new_data.loc[idx, parameters.keys()] = scores

# ─── MERGE THE SCORES BACK INTO THE ORIGINAL DATAFRAME ─────────────────────────
for col in new_data.columns:
    df[col] = new_data[col]

# ─── SAVE THE SCORES TO EXCEL ──────────────────────────────────────────────────
output_excel_path = os.path.join(BASE_PATH, "new_data.xlsx")
new_data.to_excel(output_excel_path, index=False)
print(f"Saved scored data to: {output_excel_path}")