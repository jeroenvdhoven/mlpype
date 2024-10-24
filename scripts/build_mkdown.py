import shutil
from importlib import import_module
from inspect import getmembers, getmodule, isclass, ismodule
from pathlib import Path
from typing import Dict, List

import yaml

root_path = Path(__file__).parent.parent.absolute()
doc_root = Path(root_path) / "docs"


def main() -> None:
    shutil.rmtree(doc_root, ignore_errors=True)
    autoapi_dirs = []  # location to parse for API reference
    for f in root_path.iterdir():
        if f.name.startswith("mlpype-") and f.is_dir():
            autoapi_dirs.append(f)

    print(f"Detected: {autoapi_dirs}")
    # for package_folder in [root_path / "mlpype-base"]:
    structure = {}
    for package_folder in autoapi_dirs:
        # if "sklearn"  not in str(package_folder):
        # continue
        package_root = package_folder / "mlpype"
        structure.update(_walk_package(package_root, "mlpype", skip_folders=["wheel"]))

    package_names_in_docs = sorted([f.name for f in (doc_root / "mlpype").iterdir() if f.is_dir()])

    _update_index(package_names_in_docs)
    # Update mkdocs yaml with all navs
    _update_mkdocs_yml(structure)


def _update_index(package_names_in_docs):
    package_references = "\n".join([f"- [mlpype {p}](mlpype/{p}.md)" for p in package_names_in_docs])

    with open(root_path / "README.md", "r") as f:
        readme = f.read()

    readme = (
        f"""
# MLpype\n\n
Subpackages for MLpype:
{package_references}
"""
        + readme[1:]
    )

    with open(doc_root / "index.md", "w") as f:
        f.writelines(readme)


def _update_mkdocs_yml(structure: dict) -> None:
    with open(root_path / "mkdocs.yml", "r") as f:
        contents = yaml.safe_load(f)

    new_nav: list = contents["nav"]
    for nav in new_nav:
        if "Packages" in nav:
            nav["Packages"] = _create_nav(structure, ["mlpype"])
            break
    contents["nav"] = new_nav

    with open(root_path / "mkdocs.yml", "w") as f:
        yaml.safe_dump(contents, f)


def _create_nav(structure: dict, root: list) -> list:
    result = []

    for key, value in structure.items():
        if isinstance(value, dict):
            result.append({key: _create_nav(value, root + [key])})
        else:
            if key == "__init__.py":
                name = root[-1]
                path = f"{'/'.join(root)}.md"
            else:
                name = key.replace(".py", "")
                path = f"{'/'.join(root + [name])}.md"
            result.append({name: path})

    return sorted(result, key=_get_key)


def _get_key(l: dict):
    key = list(l.keys())[0]
    if key == "__init__.py":
        return f"0{key}"
    else:
        return key


def _walk_package(root: Path, import_root: str, skip_folders: List[str]) -> dict:
    # subfolders will be walked.
    # init will provide the main doc package.
    # other python files will be imported, act as stub.
    result = {}

    for f in root.iterdir():
        new_root = root / f
        if new_root.is_dir() and f.name != "__pycache__":
            if f.name in skip_folders:
                continue

            new_import_root = f"{import_root}.{f.name}"
            print(f"Generating markdown from subpackage {new_import_root}")
            import_module(new_import_root)
            result[f.name] = _walk_package(new_root, new_import_root, skip_folders)
        elif new_root.suffix == ".py":
            new_import_root = f"{import_root}.{f.name.replace('.py', '')}"
            result[f.name] = f.name
            if f.name == "__init__.py":
                # Generate markdown from init
                print(f"Generating markdown from INIT {import_root}")
                _create_mkdown_from_file(new_import_root, True)
            else:
                # Generate markdown from python file
                print(f"Generating markdown from python file {new_import_root}")
                _create_mkdown_from_file(new_import_root, False)
    return result


def _create_mkdown_from_file(import_root: str, is_init: bool) -> None:
    contents = _list_contents_of_file(import_root, is_init)

    markdown = _create_markdown_from_contents(contents, import_root)
    file_paths = import_root.split(".")

    if file_paths[-1] == "__init__":
        docs_path = doc_root.joinpath(*file_paths[:-2])
        file_name = file_paths[-2]
    else:
        docs_path = doc_root.joinpath(*file_paths[:-1])
        file_name = file_paths[-1]

    docs_path.mkdir(parents=True, exist_ok=True)
    with open(docs_path / f"{file_name}.md", "w") as f:
        f.write(markdown)


def _list_contents_of_file(import_root: str, is_init: bool) -> Dict[str, List[str]]:
    module = import_module(import_root)

    if is_init:
        root_module_name = import_root.replace(".__init__", "")
    else:
        root_module_name = import_root

    result = {
        "classes": [
            f"{root_module_name}.{cls_name}"
            for cls_name, cls_obj in getmembers(module)
            if isclass(cls_obj)
            and (
                (is_init and "mlpype" in getmodule(cls_obj).__name__)
                or (not is_init and import_root in getmodule(cls_obj).__name__)
            )
        ],
        "functions": [
            f"{root_module_name}.{func_name}"
            for func_name, func_obj in getmembers(module)
            if callable(func_obj)
            and not isclass(func_obj)
            and (
                (is_init and "mlpype" in getmodule(func_obj).__name__)
                or (not is_init and import_root in getmodule(func_obj).__name__)
            )
            and not func_name.startswith("_")
        ],
    }

    if is_init:
        result["modules"] = [
            f"{root_module_name}.{mod_name}"
            for mod_name, mod_obj in getmembers(module)
            if ismodule(mod_obj) and "mlpype" in getmodule(mod_obj).__name__
        ]

    return result


def _create_markdown_from_contents(contents: Dict[str, List[str]], file_path: str):
    if file_path == "mlpype.sklearn.model.__init__":
        # TODO: find a good way to incorporate dynamic models.
        classes_str = ["::: mlpype.sklearn.model.SklearnModel", "::: mlpype.sklearn.model.SklearnModelBaseType"]
    else:
        classes_str = [f"::: {c}" for c in contents["classes"]]
    functions_str = [f"::: {c}" for c in contents["functions"]]

    if file_path.endswith("__init__"):
        file_path = file_path.replace(".__init__", "")

    if "modules" in contents:
        core_package = file_path.split(".")[-1]
        # Generate links
        module_names = [m.split(".")[-1] for m in contents["modules"]]
        modules_str = [f"- [{name}]({core_package}/{name}.md)" for name in module_names]
        if len(modules_str) > 0:
            modules_str = ["## Index:\n"] + modules_str
    else:
        modules_str = []

    return f"::: {file_path}\n\n" + "\n\n".join(modules_str + classes_str + functions_str)


if __name__ == "__main__":
    main()
