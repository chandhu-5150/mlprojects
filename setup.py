
from setuptools import find_packages, setup


def get_requirements(file_path):
    with open(file_path) as file_obj:
        requirements = file_obj.read().splitlines()

    return [req.replace("-e .", "") for req in requirements if req.strip()]


setup(
    name="mlprojects",
    version="0.0.1",
    author="P. Chandra Sekhar Reddy",
    author_email="chandhu2482005@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

