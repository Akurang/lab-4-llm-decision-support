#SUMMARY_PROMPT

summary_v1, usage_v1 = ask_llm(f"Summarize this:\n\n{LETTERS['L002']}\n\n{LETTERS['L006']}")

SUMMARY_PROMPT_V2_SYSTEM = """You are an assistant to a microfinance loan officer. Your job is to summarize 
loan applications into brief, factual summaries. Follow these rules strictly:
- Be factual and neutral
- Do not invent any details not in the letter
- Keep summary to 3-4 sentences total
- Focus on: applicant name, loan amount, purpose, repayment ability"""

summary_v2, usage_v2 = ask_llm(
    f"Summarize this loan application:\n\n{LETTERS['L002']}\n\n \n\n{LETTERS['L006']}",
    system_prompt=SUMMARY_PROMPT_V2_SYSTEM,
    temperature=0


#EXTRACT_PROMPT


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


json_answer,json_usage = ask_llm(
        user_prompt=f" Generate a raw JSON response for the following letter: {letter_text}. Do not include markdown code block fences.",
                        system_prompt=extract_prompt_system,temperature=0)


#BRIEF_PROMPT

BRIEF_PROMPT_system = """After receiving the letter and extracted JSON, output the following:
    1. Strengths (bullet points, grounded in the letter)
    2. Risks / red flags (bullet points)
    3. Missing information the officer should request
    4. Suggested next step (e.g. "invite for interview", "request documents",
       "flag for senior review") — NOT "approve" or "reject".

    The overall final decisions are made by humans.
    
"""


BRIEF_PROMPT_L001,BRIEF_usage_1 = ask_llm(
    user_prompt=f"Given \n\n{LETTERS['L001']} and {response_dict[0]} generate briefs using the guidlines listed in the system_prompt",
    system_prompt=BRIEF_PROMPT_system)
