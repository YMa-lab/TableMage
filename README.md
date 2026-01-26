# TableMage &nbsp; 🧙‍♂️📊

![Python Version](https://img.shields.io/badge/python-3.12-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Tests Passing](https://github.com/YMa-lab/TableMage/actions/workflows/test.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/tablemage/badge/?version=latest)](https://tablemage.readthedocs.io/en/latest/?badge=latest)


TableMage is a Python package for low-code/conversational clinical data science.
TableMage can help you quickly explore tabular datasets, 
easily perform regression analyses,
and effortlessly benchmark machine learning models.


## Installation

We recommend installing TableMage in a new virtual environment. TableMage supports Python version 3.12.

To install TableMage:
```
git clone https://github.com/YMa-lab/TableMage.git
cd TableMage
pip install .
cd ..
```

> [!NOTE]
> **For MacOS users:** You might run into an error involving [XGBoost](https://xgboost.readthedocs.io/en/stable/#), one of TableMage's dependencies, when using TableMage for the first time.
> To resolve this error, you'll need to install libomp: `brew install libomp`. This requries [Homebrew](https://brew.sh/).

## Quick start (low-code)

You'll likely use TableMage for machine learning model benchmarking. Here's how to do it.

```python
import tablemage as tm
import pandas as pd
import joblib

# load table (assume 'y' is a numeric variable we wish to predict)
df = ...

# initialize an Analyzer object
analyzer = tm.Analyzer(df, test_size=0.2)

# preprocess data, taking care to exclude the target variable 'y' from the operations
analyzer.dropna(
    include_vars=['y']
).impute(
    exclude_vars=['y']
).scale(
    exclude_vars=['y']
)

# train regressors
reg_report = analyzer.regress(  # categorical variables are automatically one-hot encoded
    models=[                    # hyperparameter tuning is preset and automatic
        tm.ml.LinearR('l2', name='ridge'),
        tm.ml.TreesR('random_forest', name='rf'),
        tm.ml.TreesR('xgboost', name='xgb'),
    ],
    target='y',                 # automatically drops examples with missing values in target variable
    predictors=None,            # None signifies all variables except target variable
    feature_selectors=[
        tm.fs.BorutaFSR()       # select subset of predictors prior to training
    ]
)

# view model metrics
print(reg_report.metrics('test'))

# predict on new data
new_df = ...
ridge_model = reg_report.model('ridge').sklearn_pipeline()
y_pred = ridge_model.predict(new_df)

# save as sklearn pipeline
joblib.dump(ridge_model, 'ridge.joblib')
```


## Quick start (conversational)

First, install the required additional dependencies.
```
pip install "tablemage[agents]"
```

Next, add your API key. You only need to do this once; your API key will be written to a local `.env` file.
```python
import tablemage as tm
tm.use_agents()                                             # import the agents module
tm.agents.set_key("openai", "add-your-api-key-here")        # set API key
```

You can open up a chat user interface by running the following code 
and navigating to the URL that appears in the terminal.
Your conversation with the ChatDA, the AI agent, appears on the left, 
while ChatDA's analyses (figures made, tables produced, TableMage commands used) 
appear on the right.

```python
import tablemage as tm
tm.use_agents()
tm.agents.options.set_llm(
    llm_type="openai", 
    model_name="gpt-4o", 
    temperature=0.1
)
# optionally, multimodal ChatDA can interpret figures
tm.agents.options.set_multimodal_llm(
    llm_type="openai",
    model_name="gpt-4o",
    temperature=0.1
)                           # multimodal LLM must be specified for multimodal ChatDA
tm.agents.ChatDA_UserInterface(
    split_seed=42,
    memory_size=500,        # higher memory_size --> higher cost but better performance
    tools_only=True,        # if tools_only=True, Python interpreter is disabled
    tool_rag_top_k=5,       # higher k --> higher cost but better performance
    multimodal=True         # if multimodal=True, ChatDA can analyze figures
).run(debug=False)
```

Or, you can chat with the AI agent directly in Python:

```python
import pandas as pd
import tablemage as tm
tm.use_agents()
tm.agents.options.set_llm(
    llm_type="openai", 
    model_name="gpt-4o", 
    temperature=0.1
)

# load table
df = ...

# initialize a ChatDA object
agent = tm.agents.ChatDA(
    df,                     # additional parameters can be set, e.g. memory type, 
    test_size=0.2           # disabling/enabling Python environment, etc.
)

# chat with the agent
print(agent.chat("Compute the summary statistics for the numeric variables."))
```

> [!NOTE]
> You must be connected to the internet to use the `agents` module, even if you are using Ollama to run a locally-hosted LLM.
> TableMage's agent, ChatDA, relies on FastEmbed for retrieval augmented generation, but it may need to download the FastEmbed model from the internet prior to use.
> ChatDA can be run with a local LLM and FastEmbed, ensuring total data privacy.

## Updates

- December 2025: We have released a preprint of our work on medRxiv!
- February 2025: We have released an alpha version of TableMage on PyPI!

## Citation

If this software was beneficial in your work, please consider citing it as follows:

```bibtex
@article{Yang2025-TableMage,
    author = {Yang, Andrew and Woo, Joshua and Zhang, Ryan and Mach, Alan and Ramkumar, Prem and Ma, Ying},
    title = {Tool-wielding language-model-based agent offers conversational exploration of clinical tabular data},
    elocation-id = {2025.12.01.25341392},
    year = {2025},
    doi = {10.64898/2025.12.01.25341392},
    publisher = {Cold Spring Harbor Laboratory Press},
    URL = {https://www.medrxiv.org/content/early/2025/12/02/2025.12.01.25341392},
    eprint = {https://www.medrxiv.org/content/early/2025/12/02/2025.12.01.25341392.full.pdf},
    journal = {medRxiv}
}
```


