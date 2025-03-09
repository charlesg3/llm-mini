# LLM Mini Project

This is a simple project for training a mini Language Learning Model (LLM) following Andrej Karpathy's tutorial video: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=7xTGNNLPyMI).

The project aims to implement a smaller version of a language model to understand the core concepts behind transformer-based models like GPT.

## Project Structure

- `main.py`: Command-line interface for various operations
- `data_retrieval.py`: Module for downloading training data
- `retrieve_assistant_data.py`: Module for downloading assistant conversation data
- `config.py`: Configuration management module
- `model.py`: Implementation of a GPT-2 style transformer model
- `training.py`: Training functionality for the model on web text data
- `fine_tune_assistant.py`: Fine-tuning functionality for the model on assistant conversations
- `data/`: Directory containing training data and cached files

## Getting Started

### Prerequisites

- Python 3.6+
- Required packages: pandas, pyarrow, requests, tqdm, tiktoken, datasets, torch

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

This command will read the text from `data/text.txt`, tokenize it using the tiktoken library with the encoding specified in your config, and save the tokens to `data/tokens.parquet`. It will also save token frequency information to `data/token_frequencies.parquet`.

Options:
- `--input`: Input text file path (default: data/text.txt)
- `--output`: Output tokens parquet file path (default: data/tokens.parquet)
- `--encoding`: Tiktoken encoding name (default: from config)

Example with options:

```bash
python main.py tokenize_data --input data/custom_data.txt --output data/custom_tokens.parquet
```

#### Training the Model

To train a GPT model on the tokenized web text data:

```bash
python main.py train
```

This command will train a GPT model on the tokenized data and save checkpoints to `checkpoints/web/`.

Options:
- `--input`: Input tokens file (default: data/tokens.parquet)
- `--output-dir`: Directory to save checkpoints (default: checkpoints/web)
- `--model-size`: Model size (micro or mini, default: micro)
- `--batch-size`: Batch size for training (default: 8)
- `--epochs`: Number of epochs to train (default: 3)
- `--save-every`: Save checkpoint every N epochs (default: 1)
- `--eval-every`: Evaluate on validation set every N epochs (default: 1)
- `--checkpoint`: Path to checkpoint to resume from
- `--sample`: Sample from the model after training
- `--prompt`: Prompt for sampling (default: "Once upon a time")
- `--max-tokens`: Maximum tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.8)

Example with options:

```bash
python main.py train --model-size mini --batch-size 4 --epochs 5 --sample
```

#### Fine-tuning on Assistant Conversations

To fine-tune a GPT model on assistant conversations:

```bash
python main.py finetune
```

This command will fine-tune a GPT model on the assistant conversation data and save checkpoints to `checkpoints/assistant/`.

Options:
- `--input`: Input assistant data file (default: data/assistant_data/assistant_data.json)
- `--output-dir`: Directory to save checkpoints (default: checkpoints/assistant)
- `--batch-size`: Batch size for training (default: 4)
- `--epochs`: Number of epochs to train (default: 5)
- `--save-every`: Save checkpoint every N epochs (default: 1)
- `--eval-every`: Evaluate on validation set every N epochs (default: 1)
- `--checkpoint`: Path to checkpoint to resume from
- `--sample`: Sample from the model after training
- `--prompt`: Prompt for sampling (default: "User: How do I implement a transformer model in PyTorch?\n\nAssistant:")
- `--max-tokens`: Maximum tokens to generate (default: 200)
- `--temperature`: Sampling temperature (default: 0.7)

Example with options:

```bash
python main.py finetune --batch-size 2 --epochs 10 --sample
```

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Andrej Karpathy for the excellent tutorial
- Hugging Face for providing the fineweb dataset
- The OpenAssistant team for the [OASST1 dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) - a human-generated, human-annotated assistant-style conversation corpus