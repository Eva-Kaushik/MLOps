import nbformat
from nbconvert import PythonExporter
from pathlib import Path

def convert_ipynb_to_py(ipynb_file, output_dir=None):
    """
    Convert a Jupyter notebook (.ipynb) to a Python script (.py).

    Parameters:
        ipynb_file (str or Path): Path to the input Jupyter notebook file.
        output_dir (str or Path, optional): Directory where the Python script will be saved.
            If not specified, the Python script will be saved in the same directory as the input notebook.

    Returns:
        str: Path to the generated Python script file.
    """
    # Load the notebook
    with open(ipynb_file, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Create a Python exporter
    exporter = PythonExporter()

    # Convert the notebook to Python script
    python_script, _ = exporter.from_notebook_node(notebook)

    # Determine the output directory
    if output_dir is None:
        output_dir = Path(ipynb_file).parent
    else:
        output_dir = Path(output_dir)

    # Determine the output file name
    output_file = output_dir / (Path(ipynb_file).stem + ".py")

    # Save the Python script
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(python_script)

    return str(output_file)

# Example usage:
if __name__ == "__main__":
    # Specify the paths to the input Jupyter notebook files
    notebook_paths = [
        "C:\\Users\\ekaushik\\Desktop\\MLOps\\Assignment 1\\prep.ipynb",
        "C:\\Users\\ekaushik\\Desktop\\MLOps\\Assignment 1\\train.ipynb"
    ]

    # Convert each notebook to Python script
    for notebook_path in notebook_paths:
        output_script_file = convert_ipynb_to_py(notebook_path)
        print(f"Jupyter notebook {notebook_path} converted to Python script: {output_script_file}")
