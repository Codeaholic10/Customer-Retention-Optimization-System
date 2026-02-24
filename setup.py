from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements.txt and return a list of package names.
    Strips the '-e .' editable install entry, which is not a real package.

    Args:
        file_path: Path to the requirements file.

    Returns:
        List of requirement strings.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        # Remove blank lines and comments
        requirements = [
            req for req in requirements if req and not req.startswith("#")
        ]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="customer_seg_churn",
    version="0.1.0",
    author="Aaditya Gautam",
    author_email="aadi.tidy99@gmail.com",
    description=(
        "An end-to-end ML project for customer segmentation "
        "and churn prediction with profit optimization."
    ),
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
)
