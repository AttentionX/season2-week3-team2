
from utils import (
    openai_call_chat_single,
    print_color,
    Colors,
    parse_json_in_str,
)
from prompt import (
    make_planning_prompt,
)
import pandas as pd
import openai
import os


DATA_EXPLANATION = {
    "air-pollution": {
        "description": "This is a dataset of air pollution of Korea durin  Jan 2021. It contains daily PM10, PM2.5, O3 levels.",
        "filepath": "Air pollution.csv",
    },
    "titanic": {
        "description": "This is a dataset of air pollution of Korea durin  Jan 2021. It contains daily PM10, PM2.5, O3 levels.",
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
    print_color(planning_prompt, Colors.GREY)
    out = openai_call_chat_single(planning_prompt)
    return parse_json_in_str(out)

plan = make_plan(df)
print_color(f"[Plan]:{plan}", Colors.GREEN)
# 2. Actually perform the analysis
[
    {
        "plan": {},
        "figure_path": "XXX.png",
        "description": "This is a figure about XXX",
    }
]
# 3. Make the result into a nice report
