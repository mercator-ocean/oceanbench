import nbformat
from nbclient import NotebookClient

notebook = nbformat.read("assets/glonet-example.ipynb", as_version=4)

client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": "assets/"}},
)

client.execute()

nbformat.write(notebook, "assets/glonet-example-report.ipynb")
