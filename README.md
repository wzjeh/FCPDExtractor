# OSSExtractor
Synthesis parameter extraction for on-surface synthesis literature.

OSSExtract extracts information from text and identifies the following parameters:
- Reaction type
- Precursor
- Temperature
- Substrate
- Product
- Dimension

OSSExtract uses a local large language model to extract on-surface reaction parameters instead of relying on an online LLM or an API. The LLM model can be downloaded from [Nous-Hermes-Llama2-GGUF](https://huggingface.co/TheBloke/Nous-Hermes-Llama2-GGUF) and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). There are also other available models to choose from.

## Using the OSSExtractor
Before starting this project, ensure that the LLM model has been downloaded from the above websites. To install the ecosystem, use the command below:
```
pip install gpt4all
```
### Text Parser
  #### Html.py
  Obtain text from websites.
  #### PDF_to_TXT.py
  Convert PDF files to TXT files and split the text into paragraphs.
  #### PDF_Processing.py 
  Remove redundant information not relevant to the text to save computing resources.

### Text Extraction
  #### Embedding_and Similarity.py
  Embed the text into a high-dimensional space matrix and calculate the cosine similarity between the paragraphs and the sample text.
  #### Filter.py
  Keep the paragraphs that contain the synthesis parameters and filter out irrelevant content.
  #### Abstract.py
  Abstract the filtered paragraphs, normalize the text, and refine the connections between information.
  #### Summerized.py
  Summarize the synthesis parameters into a table.


## Running OSSExtractor
1. Set "allow_download=False" to use the downloaded model of your choice.
2. Run the Python files in order to extract the parameters.

