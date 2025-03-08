# LLM Mini Project

This is a simple project for training a mini Language Learning Model (LLM) following Andrej Karpathy's tutorial video: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=7xTGNNLPyMI).

The project aims to implement a smaller version of a language model to understand the core concepts behind transformer-based models like GPT.

## Project Structure

- `main.py`: Command-line interface for various operations
- `data_retrieval.py`: Module for downloading training data
- `retrieve_assistant_data.py`: Module for downloading assistant conversation data
- `config.py`: Configuration management module
- `data/`: Directory containing training data and cached files

## Getting Started

### Prerequisites

- Python 3.6+
- Required packages: pandas, pyarrow, requests, tqdm, tiktoken, datasets

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

This command will download webpages from the Hugging Face fineweb dataset and store them in `data/text.txt`. The downloaded parquet files will be cached in `data/parquet_cache` to avoid re-downloading them in future runs.

Options:
- `--force`: Force overwrite of existing output file
- `--samples`: Number of samples to download (default: 2000)
- `--output`: Output file path (default: data/text.txt)
- `--cache-dir`: Directory to cache parquet files (default: data/parquet_cache)

Example with options:

```bash
python main.py retrieve_data --samples 500 --output data/custom_data.txt
```

#### Retrieving Assistant Conversation Data

To download conversation data for training an assistant model:

```bash
python main.py retrieve_assistant_data
```

This command will download the OpenAssistant Conversations Dataset (OASST1), filter for English content, and store it in `data/assistant_data/`. The data is organized into conversation threads based on parent-child relationships between messages.

Options:
- `--force`: Force overwrite of existing files
- `--output-dir`: Directory to store the downloaded data (default: data/assistant_data)
- `--lang`: Language code to filter for (default: 'en' for English)

Example with options:

```bash
python main.py retrieve_assistant_data --lang fr --output-dir data/french_assistant_data
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
- The OpenAssistant team for the [OASST1 dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) - a human-generated, human-annotated assistant-style conversation corpus