import json
from bs4 import BeautifulSoup
from pathlib import Path

models = ["glonet"]


def find_result_html(model: str) -> str:
    input_notebook = open(f"../assets/{model}_sample.report.ipynb")
    raw_notebook = json.load(input_notebook)
    for cell in raw_notebook["cells"]:
        if "oceanbench.metrics.rmse_compared_to_glorys_variables(challenger_datasets)" in cell["source"]:
            html_output = cell["outputs"][0]["data"]["text/html"]
            cleaned_html_output = "".join([line.removesuffix("\n") for line in html_output])
            return cleaned_html_output


Path("_result_tables").mkdir(parents=True, exist_ok=True)
for model in models:
    scores = {}
    result = find_result_html(model)
    soup = BeautifulSoup(result)
    tbody = soup.find("tbody")
    rows = tbody.find_all("tr")
    for row in rows:
        variable = row.find("th").string
        scores[variable] = {k: float(v.string) for k, v in enumerate(row.find_all("td"))}
    output = open(f"_result_tables/{model}.json", "w+")
    json.dump(scores, output)
