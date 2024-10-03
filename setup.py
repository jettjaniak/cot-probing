from setuptools import find_packages, setup

setup(
    name="cot_probing",
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,
)
