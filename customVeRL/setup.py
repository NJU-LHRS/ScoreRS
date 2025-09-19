import os
import re

from setuptools import find_packages, setup


def get_version() -> str:
    with open(os.path.join("verl", "__init__.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"__version__\W*=\W*\"([^\"]+)\""
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> list[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [
            line.strip()
            for line in file_content.strip().split("\n")
            if not line.startswith("#")
        ]
        return lines


extra_require = {
    "dev": ["pre-commit", "ruff"],
}


def main():
    setup(
        name="verl",
        version=get_version(),
        description="VeRL",
        long_description_content_type="text/markdown",
        author="verl",
        author_email="",
        license="Apache 2.0 License",
        url="",
        package_dir={"": "."},
        packages=find_packages(where="."),
        python_requires=">=3.9.0",
        install_requires=get_requires(),
        extras_require=extra_require,
    )


if __name__ == "__main__":
    main()
