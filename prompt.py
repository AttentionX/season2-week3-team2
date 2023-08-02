import pandas as pd

def make_data_format_str(df):
    return f"columns:\n{str(list(df.columns))}\n\nshape:\n{df.shape}\n\ndf.head():\n{str(df.head())}\n\ndf.describe():\n{str(df.describe())}"


PLANNING_PROMPT = """The following is a description of a dataset in csv format.
Suppose the dataset is currently read and saved as `df`. 
Make a list of items to analyze from the dataset. 
Format the plan as a list in the following format in between 3 backticks:
```
[
    "analysis 1",
    "analysis 2",
    "analysis 3",
]
```
Description:{description}
An overview of the dataframe:
{stats}
Analysis plan:
"""

def make_planning_prompt(df: pd.DataFrame, description: str = None):
    stats = make_data_format_str(df)
    return PLANNING_PROMPT.format(
        description=description,
        stats=stats
    )

GENERATE_CODE_PROMPT= """Create Python code to analyze the dataset in terms of the goal.
Goal: {plan}

Suppose the dataset is currently read and saved as `df`. 
Make sure to visualize the results and save the results in `./workdir/{{result_file}}` use a very specific name that won't be overlapped.
Use xticks, yticks, title, and put a legend to create a complete figure.
Use libraries such as matplotlib and seaborn. Import neccessary libraries.
Put the code in between 3 backticks as below.
```
code to visualize data
```

python code:"""


DESCRIBE_CODE_RESULT = """Describe all the files that will be created from executing the code below. 
Format the output in the following way as a json file with 3 backticks:
```
[
    {{
        \"file\": \"/path/to/file1.png\",
        \"description\": \"This is a figure about XXX.\"
    }},
    ...
]
```

Goal of code: {plan}
Code:
```
{code}
```

Description:
"""

def get_code_result_prompt(plan: str, code: str):
    # run_analysis_code(code, df)
    return DESCRIBE_CODE_RESULT.format(plan=plan, code=code)
    

MAKE_REPORT_PROMPT = """Make a report about the analysis results.
Format the prompt using latex code.
The goals was to analyze the following things:
{plan}

The following figures can be used to enhance the report.
Refer to the descriptions and use the figures in the report when describing related sections.
Figures:
```
{figures}
```
Latex report:
"""

def make_report_prompt(plan: str, figures: list[dict]):
    return MAKE_REPORT_PROMPT.format(plan=str(plan), figures=str(figures))