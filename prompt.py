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
