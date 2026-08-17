# %% [markdown]
# # Lab 4: LLMs and Prompt Engineering for Decision Support
#
# **Duration:** 2 weeks [30 Jul - 13 Aug, 2026]
# **Due Date:** 13th August, 2026
# **Format:** Jupyter Notebook / Google Colab + external APIs + GitHub version control
# **Grading:** This is a graded lab.
#
# **Student Name:** Gabriel Akurang
# **Student ID:** 35882028
#
# ---
#
# ### Objective
#
# In the previous labs you *trained* models. In this lab you will *use* a model that someone
# else spent millions of dollars training — a **Large Language Model (LLM)** — and learn that
# getting good results out of one is an engineering discipline of its own: **prompt
# engineering**.
#
# You will build a **decision support system for a microfinance loan officer**. Given a pile of
# free-text loan application letters, your system will:
#
# 1. **Summarize** each application into a short, factual brief,
# 2. **Extract** specific structured data points (JSON) that a downstream system could store,
# 3. Produce a **decision-support recommendation** — while keeping the human firmly in the loop.
#
# Just as importantly, you will **evaluate** the LLM's output for quality, reliability, and
# appropriateness: Does it hallucinate? Is it consistent across runs? Should it be trusted to
# make the final call?
#
# ---
#
# ### Choosing an API provider
#
# You need an LLM API with a **free tier**. Recommended options (pick ONE):
#
# | Provider | Free tier | Notes |
# |---|---|---|
# | **Groq** (recommended) | Yes, generous | OpenAI-compatible API, very fast, open models (Llama) |
# | **Google Gemini** | Yes | `google-generativeai` package |
# | **Hugging Face Inference API** | Yes, limited | Many open models |
# | OpenAI / Anthropic | Paid | Fine if you already have credits |
#
# The notebook's example code uses the **OpenAI-compatible chat format** (works with Groq and
# OpenAI directly; Gemini users adapt the call in one place). Everything else in the lab is
# provider-agnostic.

# %% [markdown]
# ---
# ### Part 0: Repository and API-key setup
#
# 1. Create a **public** repository named `lab-4-llm-decision-support` and save this notebook
#    inside it.
# 2. Sign up with your chosen provider and create an **API key**.
# 3. **NEVER hard-code or commit your API key.** This is a graded requirement.
#    - Locally: put it in a `.env` file and add `.env` to `.gitignore`.
#    - Colab: use the Secrets panel (key icon) and read it with `google.colab.userdata`.
# 4. Add a `requirements.txt`: `openai python-dotenv pandas matplotlib`.
# 5. Commit and push after **each Part** — we will check for incremental commits.
#
# > **A leaked key in your commit history = resubmission + penalty.** Keys can be scraped from
# > public repos within minutes.

# %%
# API-key setup — DO NOT hard-code your key in this cell.

import os

# --- Local (with a .env file) ---
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["GROQ_API_KEY"]

# --- Google Colab (Secrets panel) ---
# from google.colab import userdata
# API_KEY = userdata.get("GROQ_API_KEY")

# TODO: set API_KEY using ONE of the methods above.

# OpenAI-compatible client (works for Groq and OpenAI; Gemini users see their docs):
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",  # remove this line if using OpenAI itself
)
MODEL = "llama-3.3-70b-versatile"  # or your provider's model name

print("Client ready.")

# %% [markdown]
# ---
# # Section 1 — Talking to an LLM Programmatically
#
# Before building anything, understand the anatomy of an API call: **messages and roles**
# (`system`, `user`, `assistant`), and the **generation parameters** (`temperature`,
# `max_tokens`).

# %% [markdown]
# ### Part 1.1 — Your first API call

# %%
# TODO: Write a helper function you will reuse for the WHOLE lab:


def ask_llm(
    user_prompt,
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=500,
):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content, response.usage


#
# TODO: Call it once with a simple question and print the answer.
response, usage = ask_llm("How are you?")
print(response)

# TODO: Print response.usage as well — how many tokens did your call consume?
print()
print(usage)
print(f"\nTotal Tokens Used: {usage.total_tokens}")


# %% [markdown]
# **Student Reasoning — Anatomy of a call**
# *1. What is the difference between the `system` and `user` roles? Give an example of
# something that belongs in each.*
# *2. What is a token, roughly? Why do API providers bill per token rather than per request?*
#
# > **Answer:** 1.The system role defines the rules, behavior, and context that the model should follow, while the user role contains the actual task or question the user wants the model to perform. For example, “You are a teacher” could be a system instruction, while “What is 2 + 2?” could be a user instruction.
# 2. A token is a small unit of text that a language model processes. It can represent a whole word, part of a word, punctuation, or other text. API providers bill per token because different requests can involve very different amounts of text and computation; charging per request would not accurately reflect the amount of processing required.

# %% [markdown]
# ### Part 1.2 — Temperature: the randomness dial

# %%
# TODO: Ask the SAME question 5 times at temperature=0.0 and 5 times at temperature=1.2.
#   A good test question: "Suggest a name for a savings product for market traders in Accra."

count_0 = 0


for i in range(10):
    if count_0 < 5:
        response_same, usage_same = ask_llm(
            "Suggest a name for a savings product for market traders in Accra.",
            temperature=0,
        )
        print(f"\nResponse {i + 1} and temperature=0")
        print(response_same)
        count_0 += 1
    else:
        response_same, usage_same = ask_llm(
            "Suggest a name for a savings product for market traders in Accra.",
            temperature=1.2,
        )
        print(f"\nResponse {i + 1} and temperature=1.2")
        print(response_same)


# TODO: Print all 10 answers, grouped by temperature.


# %% [markdown]
# **Student Reasoning — Temperature**
# *What did you observe at each temperature? For the loan decision-support system you are about
# to build, which temperature regime is appropriate, and why?*
#
# > **Answer:** When the temperature is low the options are more consistent and predictable. But for the high temperature at 1.2, there is more randomness and it leads to the introduction of new name suggestions not previously mentioned. For the loan decision-support system, the temperature = 0 is more appropriate as it ensures stability and prevents variations.

# %% [markdown]
# ---
# # Section 2 — The Dataset: Loan Application Letters
#
# Run the next cell to load **six loan application letters** submitted to a (fictional)
# microfinance institution in Ghana, plus **gold-standard extraction labels** for three of them
# (you will use these for evaluation in Section 4).
#
# Read at least two letters fully before moving on — you cannot engineer prompts for text you
# have not read.

# %%
LETTERS = {
    "L001": """Dear Sir/Madam,
My name is Akosua Mensah and I have been selling provisions at Makola Market for 12 years.
I am applying for a loan of GHS 8,000 to buy a deep freezer and expand into frozen foods.
My current stall makes about GHS 900 profit each month. I have saved GHS 2,500 with your
susu scheme over the past two years and I have never missed a contribution. I can repay
GHS 450 monthly over 20 months. My sister, a teacher, will stand as my guarantor.
Thank you for considering my application.""",
    "L002": """Hello,
I am Kwame Boateng, a commercial driver in Kumasi. I need GHS 25,000 urgently to repair my
trotro engine and settle some personal debts. Business has been slow but it will surely
pick up after the festive season. I can pay back whenever the money comes. I do not have
collateral at the moment but God willing everything will be fine. Please help me quickly.""",
    "L003": """Dear Loan Committee,
I am Efua Darko, owner of Darko Fashions, a registered dressmaking business in Takoradi
(registration no. BN-2019-4482). I employ three apprentices. I request GHS 15,000 to
purchase two industrial sewing machines and fabric stock ahead of the Christmas season.
Last year my December revenue alone was GHS 22,000; monthly profit averages GHS 2,800.
I hold a fixed deposit of GHS 5,000 with GCB which I can pledge. Proposed repayment:
GHS 1,100 monthly for 15 months. Attached are my sales records for the past 18 months.""",
    "L004": """Good day,
My name is Yaw Owusu. I want a loan for my poultry farm at Nsawam. The amount is GHS 12,000
for feed and 500 new layers. I started the farm last year. Sometimes I make good money,
around GHS 1,500 in a good month, but bird flu affected us in March and I lost many birds.
I am rebuilding now. I can repay in 18 months. My uncle has agreed to guarantee the loan
with his taxi.""",
    "L005": """Dear Manager,
I am writing on behalf of the Adenta Women's Weaving Cooperative (14 members). We seek
GHS 30,000 to buy a bulk order of yarn directly from the factory, cutting out middlemen and
raising our margins from 15% to about 35%. The cooperative has operated for 6 years and
holds GHS 9,000 in our group account. We propose repayment of GHS 2,000 monthly over
16 months, backed by our group savings and joint liability agreement.""",
    "L006": """Hi,
This is Kofi. I saw your advert. I want GHS 50,000 to start a car washing business, a
provision shop, and also import phones from Dubai. I am 22 and full of energy. I have not
started any of these yet but my friends say I am very business minded. I will pay back in
one year when the businesses are booming. No collateral but I am trustworthy.""",
}

# Gold-standard labels for three letters (for Section 4 evaluation):
GOLD = {
    "L001": {
        "applicant_name": "Akosua Mensah",
        "amount_ghs": 8000,
        "purpose": "buy deep freezer / expand into frozen foods",
        "monthly_profit_ghs": 900,
        "has_collateral_or_guarantor": True,
        "repayment_months": 20,
    },
    "L003": {
        "applicant_name": "Efua Darko",
        "amount_ghs": 15000,
        "purpose": "industrial sewing machines and fabric stock",
        "monthly_profit_ghs": 2800,
        "has_collateral_or_guarantor": True,
        "repayment_months": 15,
    },
    "L006": {
        "applicant_name": "Kofi",
        "amount_ghs": 50000,
        "purpose": "car wash, provision shop, phone imports",
        "monthly_profit_ghs": None,
        "has_collateral_or_guarantor": False,
        "repayment_months": 12,
    },
}

print(f"{len(LETTERS)} letters loaded.")

# %% [markdown]
# ---
# # Section 3 — Prompt Engineering for the Decision Support System
#
# You will now build the three components of the system, iterating on your prompts as you go.
# **Keep every major prompt version** — Section 3.4 asks you to commit your prompt templates
# and document how they evolved.

# %% [markdown]
# ### Part 3.1 — Component 1: Summarization
# Turn a rambling letter into a 3-4 sentence factual brief a busy loan officer can scan.

# %%
# TODO: Write SUMMARY_PROMPT_V1 — your first, naive attempt (e.g. just "Summarize this:").
#   Run it on L002 and L006. Read the output critically.

summary_v1, usage_v1 = ask_llm(
    f"Summarize this:\n\n{LETTERS['L002']}\n\n{LETTERS['L006']}"
)
print("=== V1 Output ===")
print(summary_v1)

# TODO: Now write SUMMARY_PROMPT_V2 as a proper template with:
#   - a system prompt giving the LLM a ROLE (e.g. "You are an assistant to a microfinance
#     loan officer...") and constraints (factual, neutral, no invented details, 3-4 sentences)
#   - a user prompt template like: f"Summarize this loan application:\n\n{letter_text}"
#   Run V2 on the same two letters at temperature=0.

SUMMARY_PROMPT_V2_SYSTEM = """You are an assistant to a microfinance loan officer. Your job is to summarize 
loan applications into brief, factual summaries. Follow these rules strictly:
- Be factual and neutral
- Do not invent any details not in the letter
- Keep summary to 3-4 sentences total
- Focus on: applicant name, loan amount, purpose, repayment ability"""

summary_v2, usage_v2 = ask_llm(
    f"Summarize this loan application:\n\n{LETTERS['L002']}\n\n \n\n{LETTERS['L006']}",
    system_prompt=SUMMARY_PROMPT_V2_SYSTEM,
    temperature=0,
)
print("\n=== V2 Output ===")
print(summary_v2)

# TODO: Compare V1 vs V2 outputs side by side. Keep both prompt versions in this notebook.


# %% [markdown]
# **Student Reasoning — Summarization prompts**
# *1. What concrete problems did V1's output have that V2 fixed? Quote examples.*
# *2. Why is "no invented details" an essential instruction in this application? What is this
# failure mode called in the LLM literature?*
#
# > **Answer:** 1. V1 is more broad and loosely indicates things which were not clearly stated but could be implied. For instance it says both are trustworthy but only Kofi in L006 stated outright that he is trustworthy. V2 makes use of relevant facts only and indicating what was saidin the letter and alsowhat they did not talk about it. For instance, it states clearly the payment plan for both Kofi and Kwame.
# 2. “No invented details” is essential because a loan decision-support system must base its assessment only on information provided by the applicants. If the model invents or assumes information, it could lead to an unfair or incorrect loan decision and this is known as hallucination.

# %% [markdown]
# ### Part 3.2 — Component 2: Structured extraction (JSON)
# Downstream software cannot read prose. Extract the fields in `GOLD` as strict JSON.

# %%
# TODO: Write EXTRACT_PROMPT —
# a template that instructs the model to return ONLY a JSON
#   object with EXACTLY these keys:
#     applicant_name (string), amount_ghs (number), purpose (string),
#     monthly_profit_ghs (number or null), has_collateral_or_guarantor (boolean),
#     repayment_months (number or null)
#   Techniques to use:
#     - explicit schema in the prompt
#     - ONE worked example (few-shot) using a letter you write yourself (not from LETTERS!)
#     - "If a field is not stated in the letter, use null. Do not guess."
# - temperature=0
import json
import pandas as pd


extract_prompt_system = """  
This is a template guide to return ONLY a JSON given a letter from thenuser prompt
  object with EXACTLY these keys:
    {applicant_name: (string),
    amount_ghs :(number), 
    purpose :(string),
    monthly_profit_ghs: (number or null), 
    has_collateral_or_guarantor: (boolean),
    repayment_months: (number or null)
    }
  
For example given a sample letter:

letter_test= 'My name is Kwame Mike and I want 400,000GHS to invest it in a startup. This is likely to generate monthly profit of 50000 GHS
                and I plan to payback 20000 GHS per month for 20 months until I successfully pay off my debt with interest. I will b using my house situated on
                the moon plaza as collateral '

Sample JSONN response to the sample letter : {
"applicant_name": "Kwame Mike",
"amount_ghs": 400000,
"purpose": "To invest in a startup",
"monthly_profit_ghs": 50000,
"has_collateral_or_guarantor": No,
"repayment_months":20
}
    

- "If a field is not stated in the letter, use null. Do not guess."
"""


# TODO: Write extract_fields(letter_text) that calls the LLM, strips any ```json fences,
#   json.loads() the result, and returns a dict. Handle parse failures gracefully
#   (return None and print a warning).


def extract_fields(letter_text, temp=0):

    json_answer, json_usage = ask_llm(
        user_prompt=f" Generate a raw JSON response for the following letter: {letter_text}. Do not include markdown code block fences.",
        system_prompt=extract_prompt_system,
        temperature=temp,
    )

    result = json.loads(json_answer)

    return result


# TODO: Run it on ALL SIX letters; collect results into a pandas DataFrame (one row per
#   letter) and display it.

response_dict = []

for letter in LETTERS:
    response_dict.append(extract_fields(LETTERS[letter]))


df = pd.DataFrame(response_dict)

print(df)


# %% [markdown]
# **Student Reasoning — Structured extraction**
# *1. Why must the few-shot example NOT come from the six letters you are processing?*
# *2. Why "use null, do not guess" — what did the model do without that instruction?*
# *3. Why is temperature=0 the right choice for extraction but arguably not for creative tasks?*
#
# > **Answer:** 1. This is it would leak information from the test data into the prompt. The model could use information it has already seen rather than independently extracting the correct information.
# 2. It is likely to fill up the null with its own answer. In this case, when the instruction was removed the output was still the same.
# 3. It is good for extraction because it has stability and maintains consistency. Not recommended for creaative tasks since it lacks randomness.

# %% [markdown]
# ### Part 3.3 — Component 3: The decision-support brief
# Combine everything: for each letter, produce a recommendation brief for the loan officer —
# strengths, risks, missing information, and a suggested next step. The system must
# **support** the decision, not **make** it.

# %%
# TODO: Write BRIEF_PROMPT — it receives the letter AND your extracted JSON, and must output:
# 1. Strengths (bullet points, grounded in the letter)
# 2. Risks / red flags (bullet points)
# 3. Missing information the officer should request
# 4. Suggested next step (e.g. "invite for interview", "request documents",
#    "flag for senior review") — NOT "approve" or "reject".
#   Give the model an explicit instruction that final decisions are made by humans.

BRIEF_PROMPT_system = """After receiving the letter and extracted JSON, output the following:
    1. Strengths (bullet points, grounded in the letter)
    2. Risks / red flags (bullet points)
    3. Missing information the officer should request
    4. Suggested next step (e.g. "invite for interview", "request documents",
       "flag for senior review") — NOT "approve" or "reject".

    The overall final decisions are made by humans.
    
"""


# TODO: Generate briefs for ALL SIX letters. Print the briefs for L001, L002, and L006 —
#   three very different applications.

BRIEF_PROMPT_L001, BRIEF_usage_1 = ask_llm(
    user_prompt=f"Given \n\n{LETTERS['L001']} and {response_dict[0]} generate briefs using the guidlines listed in the system_prompt",
    system_prompt=BRIEF_PROMPT_system,
)
BRIEF_PROMPT_L002, BRIEF_usage_2 = ask_llm(
    user_prompt=f"Given \n\n{LETTERS['L002']} and {response_dict[1]} generate briefs using the guidlines listed in the system_prompt",
    system_prompt=BRIEF_PROMPT_system,
)
BRIEF_PROMPT_L003, BRIEF_usage_3 = ask_llm(
    user_prompt=f"Given \n\n{LETTERS['L003']} and {response_dict[2]} generate briefs using the guidlines listed in the system_prompt",
    system_prompt=BRIEF_PROMPT_system,
)
BRIEF_PROMPT_L004, BRIEF_usage_4 = ask_llm(
    user_prompt=f"Given \n\n{LETTERS['L004']} and {response_dict[3]} generate briefs using the guidlines listed in the system_prompt",
    system_prompt=BRIEF_PROMPT_system,
)
BRIEF_PROMPT_L005, BRIEF_usage_5 = ask_llm(
    user_prompt=f"Given \n\n{LETTERS['L005']} and {response_dict[4]} generate briefs using the guidlines listed in the system_prompt",
    system_prompt=BRIEF_PROMPT_system,
)
BRIEF_PROMPT_L006, BRIEF_usage_5 = ask_llm(
    user_prompt=f"Given \n\n{LETTERS['L006']} and {response_dict[5]} generate briefs using the guidlines listed in the system_prompt",
    system_prompt=BRIEF_PROMPT_system,
)

print("\n===== BRIEF_PROMPT_L001 =====")
print(BRIEF_PROMPT_L001)

print("\n===== BRIEF_PROMPT_L002 =====")
print(BRIEF_PROMPT_L002)

print("\n===== BRIEF_PROMPT_L006 =====")
print(BRIEF_PROMPT_L006)

print("\n===== BRIEF_PROMPT_L003 =====")
print(BRIEF_PROMPT_L003)

# %% [markdown]
# **Student Reasoning — Decision support**
# *1. Compare the briefs for L003 (strong application) and L006 (weak application). Did the
# system identify the right strengths and red flags in each?*
# *2. Why did we forbid the model from outputting "approve"/"reject"? Give one practical and
# one ethical reason.*
#
# > **Answer:** 1. Yes, the system identified the right strenghts and red flags for each and gave suggested neext steps to better gain more info about the applicants.
# 2. There should be a human to review such a decision before the model informs the applicant. Practically, this allows a human to review the applicant's information and verify the model's assessment before making the final decision. Ethically, it reduces the risk of bias or unfair discrimination affecting someone's access to credit and ensures that the AI is used as decision support rather than as the final authority.

# %% [markdown]
# ### Part 3.4 — Commit your prompt templates
# Prompts ARE code. Save your final `SUMMARY_PROMPT`, `EXTRACT_PROMPT`, and `BRIEF_PROMPT` into
# a separate file `prompts.py` (or `prompts.md`) in your repository and commit it with a
# message describing how the prompts evolved. Paste your commit hash below.
#
# > **Commit hash:** commit eb41ccca343468f10ce19c4882654cc9f7335b53

# %% [markdown]
# ---
# # Section 4 — Evaluation: Quality, Reliability, Appropriateness
#
# An impressive demo is not a trustworthy system. Now measure it.

# %% [markdown]
# ### Part 4.1 — Extraction accuracy against gold labels

# %%
import numpy as np
from difflib import SequenceMatcher

# TODO: For the three letters in GOLD, compare your extracted DataFrame to the gold values
#   field by field. Compute per-field accuracy across the three letters
#   (name matching can be case-insensitive; numbers must match exactly).


attributes = [
    "applicant_name",
    "amount_ghs",
    "purpose",
    "monthly_profit_ghs",
    "has_collateral_or_guarantor",
    "repayment_months",
]

letter_rows = {
    "L001": 0,
    "L003": 2,
    "L006": 5,
}


def accuracy(row_idx, letter_id):
    scores = {}

    for feature in attributes:
        gold_value = GOLD[letter_id][feature]
        pred_value = df.iloc[row_idx][feature]

        if feature in ["applicant_name", "purpose"]:
            gold_text = str(gold_value).strip().lower()
            pred_text = "" if pd.isna(pred_value) else str(pred_value).strip().lower()
            # scores[feature] = 100.0 if gold_text == pred_text else 0.0
            scores[feature] = round(
                SequenceMatcher(None, gold_text, pred_text).ratio() * 100, 2
            )

        elif gold_value is None or pd.isna(pred_value):
            scores[feature] = (
                100.0 if gold_value is None and pd.isna(pred_value) else 0.0
            )

        elif isinstance(gold_value, bool):
            scores[feature] = 100.0 if bool(gold_value) == bool(pred_value) else 0.0

        else:
            scores[feature] = 100.0 if float(gold_value) == float(pred_value) else 0.0

    return scores


comparison = {
    letter_id: accuracy(row_idx, letter_id)
    for letter_id, row_idx in letter_rows.items()
}

accuracy_df = pd.DataFrame(comparison).loc[attributes]
accuracy_df["accuracy"] = accuracy_df.mean(axis=1).round(2)

print(accuracy_df)

# %% [markdown]
# ### Part 4.2 — Reliability: is the system consistent?

# %%
import json
# TODO: Run extract_fields() on letter L004 FIVE times at temperature=0 and FIVE times at
#   temperature=1.0.

results_0 = []
results_1 = []


for i in range(5):
    results_0.append(extract_fields(LETTERS["L004"], temp=0))
    print(results_0[i])


for i in range(5):
    results_1.append(extract_fields(LETTERS["L004"], temp=1.0))
    print(results_1[i])


# TODO: For each temperature, report how many of the 5 runs produced (a) valid JSON and
#   (b) identical values across runs. A simple approach: json.dumps(result, sort_keys=True)
#   and count unique strings.


def analyze_results(results):
    valid_results = []

    for result in results:
        try:
            if isinstance(result, dict):
                valid_results.append(result)
            else:
                valid_results.append(json.loads(result))
        except (json.JSONDecodeError, TypeError):
            pass

    unique_outputs = {json.dumps(result, sort_keys=True) for result in valid_results}

    return len(valid_results), len(unique_outputs)


valid_0, unique_0 = analyze_results(results_0)
valid_1, unique_1 = analyze_results(results_1)

print("\nTemperature = 0")
print("Valid JSON:", valid_0, "/ 5")
print("Unique outputs:", unique_0)

print("\nTemperature = 1.0")
print("Valid JSON:", valid_1, "/ 5")
print("Unique outputs:", unique_1)


# %% [markdown]
# ### Part 4.3 — Hallucination probing

# %%
# TODO: Design TWO adversarial tests and run them:
#   Test 1 — Ask your summarizer a question about a detail that is NOT in a letter
#     (e.g. "What is the applicant's credit score?"). Does it admit the information is
#     absent, or does it invent one?
#   Test 2 — Feed your extractor an EMPTY or IRRELEVANT text (e.g. a weather report).
#     Does it return nulls, or does it fabricate an applicant?

test1, usage_test1 = ask_llm(f"What is the color of the car in \n\n{LETTERS['L002']}")
text = (
    "Today in the Akwapem South region, the weather is mostly cloudy and mild with temperatures holding near 28°C (82°F). "
    "Skies remain overcast tonight with a low near 24°C (75°F). Winds blow gently from the southwest at around 14 to 17 km/h. "
    "The chance of rain remains very low for the rest of the night. More humid and unsettled conditions with upcoming showers are expected tomorrow."
)
test2, usage_test2 = ask_llm(
    user_prompt=f" Generate a raw JSON response for the following text: {text}. Do not include markdown code block fences.",
    system_prompt=extract_prompt_system,
    temperature=0,
)


print("====== Results for Test 1 ======\n")
print(test1)


print("\n====== Results for Test 2 ======\n")
print(test2)


# TODO: Record the outputs verbatim below and label each PASS or FAIL.

print("\nTest 1: PASS")
print("Test 2: PASS")

# %% [markdown]
# **Student Reasoning — Evaluation results**
# *1. Report your extraction accuracy. Which field was hardest for the model and why?*
# *2. What did the reliability experiment show about temperature and production systems?*
# *3. Did your system hallucinate under probing? If yes, how could the prompt (or the system
# design around it) reduce the risk?*
#
# > **Answer:** The hardest field was the purpose field as it had more texts and there was a possibility that the 2outputs were saying the same thing but because the wording was different, the comparison score reduced. Since it is mainly based on syntax and not semantics.
# 2. The reliability experiment showed that increasing the temperature generally produced more unique outputs because of increased randomness. Lower temperatures, especially temperature 0, produced more consistent and predictable results.
# 3. No the system did not hallucinate. It passed for both test.

# %% [markdown]
# ### Part 4.4 — Appropriateness: should this system exist?
# No code in this part — just judgment, which is the scarcest skill in AI for business.

# %% [markdown]
# **Student Reasoning — Appropriateness**
# *1. Letters L002 and L006 would likely be declined. If the bank fully automated decisions
# with your system, who could be unfairly harmed, and how? Consider applicants who write
# poorly in English but run solid businesses.*
# *2. Loan letters contain personal data. What are the implications of sending them to a
# third-party API in another country? What would you check before deploying this at a real
# Ghanaian microfinance institution?*
# *3. Name TWO concrete safeguards you would build around this system in production (think:
# human review points, logging, appeal processes, monitoring).*
#
# > **Answer:** 1.If the bank fully automated the decisions, applicants who write poorly in English could be unfairly harmed because the model may interpret unclear grammar, spelling, or wording as a sign of poor creditworthiness.
# 2. Considerations include data privacy, security, and legal risks. Before deploying, I would ensure it is complying with the Ghana data laws and make sure to do my due diligence on the API provider's data protection and retention policies.
# 3. The safeguards include human review and monitoring.

# %% [markdown]
# ---
# # Section 5 — Reflection
#
# *Answer in a few sentences each:*
#
# 1. **Prompting as engineering:** How is iterating on a prompt similar to and different from
#    iterating on the model hyperparameters you tuned in Lab 3?
# 2. **Trust:** After your Section 4 evaluation, would you trust this system to run unattended?
#    What single evaluation result most influenced your answer?
# 3. **Cost and scale:** Estimate (from your `response.usage` numbers) the tokens needed to
#    process 1,000 applications per month. What does that imply for provider choice?
# 4. **Looking back at the course:** You have now used classical ML (Lab 2), trained neural
#    networks (Lab 3), and used a foundation model via API (Lab 4). For a task like this one,
#    why does calling an API beat training your own model — and when would it not?
#
# > **Answer:** 1. Iterating in both cases are quite similar. For lab 3, it involved adjusting the learning rate and lab 4, making changes to the temperature to finnd the better performing netwoork and get improved responses respectively. With pronpt engineeering, the notable difference is mainly tweaaking your words or adjusting rules in the input section.
#
# 2. Not at all. This model still needs a human to review the outputs and make a final decision. Based on 4.1, the system cannnot be said to be consistent as it fgailed to achieve 100% for the purpose feature. A crucial aaspct in determining what the money will be used for.
#
# 3. 500x 1000 = 500,000. This means that provider choice becomes important as the number of applications increases.
#
# 4. This is because no training is required as training is already done and the model has the ability to process natural laanguage. It would be. best to use your own model when it has to dowith data privacy or cost at scale or developing a custom model.

# %% [markdown]
# ---
# ### Submission checklist
#
# - [ ] All cells run top-to-bottom with no errors (`Kernel -> Restart & Run All`).
# - [ ] **No API key anywhere in the notebook or the commit history.**
# - [ ] Every **Student Reasoning** box is filled in with full sentences.
# - [ ] `prompts.py` / `prompts.md` committed with your final prompt templates.
# - [ ] Evaluation tables and adversarial test outputs visible in the saved notebook.
# - [ ] Notebook pushed to `lab-4-llm-decision-support` with incremental commits.
# - [ ] Repository link submitted to the course portal.
# - [ ] AI Declaration form in Repository.
