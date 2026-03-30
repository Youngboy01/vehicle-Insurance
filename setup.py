from setuptools import setup, find_packages

setup(
    name="vehicle_insurance",
    version="0.0.1",
    author="Your Name",
    author_email="your_email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
