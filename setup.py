from setuptools import setup, find_packages

setup(
    name="dca_processor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "psycopg2-binary>=2.9.0",
        "requests>=2.26.0",
        "base58>=2.1.0",
        "solana==0.36.1",
        "streamlit",
        "plotly",
        "pandas",
        "openpyxl",
        "tqdm",
        "google-cloud-aiplatform"
    ],
    python_requires=">=3.8",
)