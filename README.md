# LLM Mini Project

This is a simple project for training a mini Language Learning Model (LLM) following Andrej Karpathy's tutorial video: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=7xTGNNLPyMI).

The project aims to implement a smaller version of a language model to understand the core concepts behind transformer-based models like GPT.

## Project Structure

- `main.py`: Command-line interface for various operations
- `data_retrieval.py`: Module for downloading training data
- `data/`: Directory containing training data and cached files

## Getting Started

### Prerequisites

- Python 3.6+
- Required packages: pandas, pyarrow, requests, tqdm, tiktoken

Install the required packages using the requirements.txt file:

```bash
pip3 install -r requirements.txt
```

### Using the main.py Program

The `main.py` script provides a command-line interface for various operations related to the LLM mini project.

#### Retrieving Training Data

The first step is to download the training data. Run:

```bash
python main.py retrieve_data
```

This command will download 200 webpages from the Hugging Face fineweb dataset and store them in `data/text.txt`. The downloaded parquet files will be cached in `data/parquet_cache` to avoid re-downloading them in future runs.

Options:
- `--force`: Force overwrite of existing output file
- `--samples`: Number of samples to download (default: 200)
- `--output`: Output file path (default: data/text.txt)
- `--cache-dir`: Directory to cache parquet files (default: data/parquet_cache)

Example with options:

```bash
python main.py retrieve_data --samples 500 --output data/custom_data.txt
```

#### Tokenizing the Data

After downloading the text data, you need to tokenize it for model training:

```bash
python main.py tokenize_data
```

This command will read the text from `data/text.txt`, tokenize it using the tiktoken library with the `cl100k_base` encoding, and save the tokens to `data/tokens.parquet`. It will also save token frequency information to `data/token_frequencies.parquet`.

Options:
- `--input`: Input text file path (default: data/text.txt)
- `--output`: Output tokens parquet file path (default: data/tokens.parquet)
- `--encoding`: Tiktoken encoding name (default: cl100k_base)

Example with options:

```bash
python main.py tokenize_data --input data/custom_data.txt --output data/custom_tokens.parquet
```

## Training the Model

After downloading the data, you can proceed with training the model following Andrej Karpathy's tutorial.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Andrej Karpathy for the excellent tutorial
- Hugging Face for providing the fineweb dataset