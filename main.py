
from utils import (
    openai_call_chat_single,
    print_color,
    Colors,
    parse_json_in_str,
)
from prompt import (
    make_planning_prompt,
    make_data_format_str,
    GENERATE_CODE_PROMPT,
    get_code_result_prompt,
    make_report_prompt,
)
import pandas as pd
import openai
import os


DATA_EXPLANATION = {
    "air-pollution": {
        "description": "This is a dataset of air pollution of Korea during Jan 2021. It contains daily PM10, PM2.5, O3 levels.",
        "filepath": "Air pollution.csv",
    },
    "titanic": {
        "description": "This is a dataset of properties of Titanic passengers and whether they survived.",
        "filepath": "titanic.csv",
    },
}["air-pollution"]

out = openai_call_chat_single("hi")
print(out)

# 1. Find points to analyze the data
data_path = os.path.join("data", DATA_EXPLANATION["filepath"])
df = pd.read_csv(data_path)

def make_plan(df):
    planning_prompt = make_planning_prompt(df, description=DATA_EXPLANATION["description"])
    out = openai_call_chat_single(planning_prompt)
    return parse_json_in_str(out)

plan_list = make_plan(df)
print_color(f"[Plan]:{plan_list}", Colors.GREEN)
# 2. Actually perform the analysis

def make_code(plan: str):
    code = openai_call_chat_single(
        "Description:\n" + make_data_format_str(df) + "\n\n" + GENERATE_CODE_PROMPT.format(plan=plan))
    try:
        code = code.replace("```python", "```")
        
        return code.split("```")[1]
    except:
        print("Error during code generation")
        return ""

def perform_analysis(plan: str, df: pd.DataFrame):
    code = make_code(plan)
    print_color("[Plan]: "+plan, Colors.GREY)
    print_color("[Generated code]:"+code, Colors.GREY)
    
    code = "import matplotlib.pyplot as plt\nplt.ion()\n" + code
    exec(code)
    
    get_code_result_p = get_code_result_prompt(plan, code)
    out = openai_call_chat_single(get_code_result_p)
    out = parse_json_in_str(out)
    
    # check format
    success = []
    if not isinstance(out, list):
        return []
    for item in out:
        if not isinstance(item, dict):
            continue
        if sorted(list(item.keys())) != ["description", "file"]:
            continue
        if not os.path.exists(item["file"]):
            continue
        success.append(item)

    print_color(f"[Analysis success]: {plan}, Out: {success}", Colors.GREEN)
    return success

analysis_result = []
for p in plan_list:
    analysis_result += perform_analysis(p, df)

print_color(f"[Analysis]:{analysis_result}", Colors.GREEN)


# 3. Make the result into a nice report
def make_report(analysis_result: list):
    report_prompt = make_report_prompt(plan_list, analysis_result)
    out = openai_call_chat_single(report_prompt)
    return out

report = make_report(analysis_result)

# TODO export to latex
with open('filename.txt', 'w') as f:
    f.write(report)
