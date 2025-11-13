# DocBlip

An agentic model which is used to improve InstructBLIPs text reading capabilities.

## Diagram of the proposed model

![DocBlip Flowchart](flowchart.png)

## Detailed instructions to run the code

### Dependencies
This project requires Python 3. You can install the necessary packages using pip:
```bash
pip install torch transformers Pillow pytesseract gradio bitsandbytes accelerate
```

You also need to have Tesseract OCR installed on your system.

**On Windows:**
You can use Chocolatey:
```powershell
choco install tesseract-ocr -y
```
Make sure to add the Tesseract installation directory to your system's PATH or specify it in the script.

**On Debian/Ubuntu:**
```bash
sudo apt-get install tesseract-ocr
```

### Running the code
To run the application, execute the main Python script:
```bash
python v1.py
```
This will start a Gradio server. You can access the interface by navigating to the local URL provided in the terminal.

*(Note: As per the instructions, submitting a .ipynb file is preferred. You can create a Jupyter Notebook that walks through the steps of the `iterative_doc_agent` function for a more detailed explanation of the process.)*
