# AI Model Ablater
A program that allows you to ablate and inspect AI models easily
  - I'm envisioning that this would render the value of all the nodes upon giving a model a prompt, and then allow you to weaken the weights that result in the nodes being "activated"

## **NOTE: Currently, this has only been tested on Llama 3.2-3B-Instruct**

## Setup
Guide to installing repository and required packages

### Prerequisites
- Python 3.13.2

### Project Installation
- Clone repository

  ```bash
  git clone https://github.com/EricL521/model-ablater.git
  ```
- Enter newly created folder
  
  ```bash
  cd model-ablater
  ```
- Create Python virtual environment
  
  ```bash
  python -m venv .venv
  ```
<a name="python-venv"></a>
- Activate Python virtual environment
  - Linux
    
    ```bash
    source .venv/bin/activate
    ```
  - Windows
    
    ```cmd
    .venv\Scripts\activate
    ```
- Download packages
  
  ```bash
  pip install -r requirements.txt
  ```

## Example Usage
### Prerequisites
- [Activate virtual environment](#python-venv)

### Installing Models
  - Hugging Face model will be installed into `./model/model_name`
  ```bash
  python install.py --model-id "meta-llama/Llama-3.2-3B-Instruct"
  ```
### Generating Images
NOTE: Add `-h` option to any script for more info
  - Generate the tensors from a sample text
  ```bash
  python get_tensors.py
  ```
  - Generate mappings (optional)
  ```bash
  python gen_mappings.py
  ```
  - Generate images
  ```bash
  python gen_images.py
  ```
  - View activations (click on pixels to select them)
  ```bash
  python view_activations.py
  ```
  - Run ablated model (selected pixels will be set to 0)
  ```bash
  python test_model.py
  ```