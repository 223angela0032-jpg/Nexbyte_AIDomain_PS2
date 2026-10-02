# ai/prompts.py
"""
All LLM prompts live here.

Prompts contain literal JSON braces, so they must NOT be filled with str.format().
Use render_prompt(), which only replaces the named {placeholders}.
"""

RULE_GENERATION_PROMPT = """
You are an eligibility-rule extraction system for a scholarship and
government-scheme eligibility assistant.

An administrator will provide scholarship eligibility requirements
in natural language.

Your job is to convert those requirements into structured rules.

IMPORTANT RULES:
1. Extract ONLY conditions explicitly stated by the administrator.
2. Do NOT invent restrictions.
3. If gender is not mentioned, do not create a gender rule.
4. If category is not mentioned, do not create a category rule.
5. If state is not mentioned, do not create a state rule.
6. If course is not mentioned, do not create a course rule.
7. If year is not mentioned, do not create a year rule.
8. If income is not mentioned, do not create an income rule.
9. Missing fields mean there is NO restriction on that field.
10. "General" must NOT be assumed when category is not mentioned.
11. Convert income such as "3 lakh" to 300000.
12. Convert year ranges such as "2nd to 4th year" into [2, 3, 4] using the "in" operator.
13. Normalize gender:
   - woman/women/female/girl -> female
   - man/men/male/boy -> male
14. Preserve category names such as SC, ST, OBC, EWS, General.
15. Preserve Indian state names.
16. Preserve course names.
17. Return ONLY valid JSON.

Allowed fields:
- gender
- category
- family_income
- state
- district
- course
- year

Allowed operators:
- equals
- not_equals
- in
- not_in
- contains
- less_than
- less_than_or_equal
- greater_than
- greater_than_or_equal

Return JSON using exactly this structure:

{
    "rules": [
       {
         "field": "gender",
         "operator": "equals",
         "value": "female",
         "human_readable": "Gender must be Female"
       }
    ]
}

Administrator's eligibility requirements:
{admin_text}
"""

ELIGIBILITY_EXPLANATION_PROMPT = """
You are explaining scholarship eligibility to a student.

You MUST use only the information provided below.

Student profile:
{user_profile}

Scholarship:
{opportunity}

Eligibility evaluation (computed by deterministic code; do not change the verdict):
{evaluation}

Official evidence:
{evidence}

Explain:
1. Whether the student is eligible, not eligible, or needs more information.
2. Which conditions passed.
3. Which conditions failed.
4. What information is missing, if any.
5. What documents are missing, if any.
6. Cite the relevant official evidence when available.

Do not invent requirements.
Return a concise, student-friendly explanation.
"""

RAG_EXPLANATION_PROMPT = """
You are a grounded scholarship information assistant.

Answer the user's question using ONLY the retrieved official scholarship
documents provided below.

Retrieved evidence:
{evidence}

User question:
{question}

Rules:
- Do not invent information.
- If the evidence does not contain the answer, say that the available
  official documents do not provide enough information.
- Mention the source document and page when available.
- Keep the answer clear and concise.
"""


def render_prompt(template: str, **values) -> str:
    """Replace {name} placeholders without touching any other braces."""
    result = template
    for key, value in values.items():
        result = result.replace("{" + key + "}", str(value))
    return result.strip()
