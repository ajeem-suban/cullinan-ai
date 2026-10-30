\# CULLINAN 💎



\### A Local AI Project for Facial Recognition and Personalized Image Generation



CULLINAN is an experimental local AI project focused on building a privacy-conscious facial analysis system, with the long-term goal of enabling personalized image generation using a person's facial identity.



The project is being developed with an emphasis on local execution, modular architecture, and efficient use of hardware resources.



> \*\*Current stage:\*\* Face engine initialization, face detection pipeline, and embedding-comparison utilities are implemented. Real-world recognition accuracy is not yet validated.



\## 🎯 Project Goals



\- Build a local facial analysis and recognition system.

\- Detect faces in images using a pretrained deep-learning model.

\- Extract facial embeddings for identity comparison.

\- Evaluate recognition performance using real images.

\- Explore personalized image generation using facial identity.

\- Keep processing local wherever practical to reduce dependence on external AI APIs.



\## ✨ Implemented Features



| Feature | Status |

|---|---|

| InsightFace engine initialization | Implemented |

| Face detection in images | Implemented |

| Facial embedding extraction | Implemented |

| Cosine similarity comparison | Implemented |

| Configurable identity-match threshold | Implemented |

| Batch processing of image directories | Implemented |

| Annotated image output | Implemented |

| Automated unit tests for comparison utilities | Implemented |

| Real-world recognition evaluation | Pending |

| Personalized image generation | Planned |



\## 🧠 Technology Stack



\- \*\*Python\*\* — Core development

\- \*\*InsightFace\*\* — Face analysis and embedding extraction

\- \*\*ONNX Runtime\*\* — Model inference

\- \*\*OpenCV\*\* — Image loading, processing, and annotation

\- \*\*NumPy\*\* — Numerical operations and embedding comparison

\- \*\*Unittest\*\* — Automated testing



\## 🏗️ Project Structure



```text

PERSONAL-AI/

│

├── photos/

│   ├── reference/       # Reference images (local only)

│   └── test/            # Test images (local only)

│

├── output/              # Generated annotated images

│

├── src/

│   └── personal\_ai/

│       ├── \_\_init\_\_.py

│       ├── config.py

│       ├── face\_engine.py

│       ├── face\_detection.py

│       └── face\_recognition.py

│

├── tests/

│   └── test\_engine.py

│

├── test\_face.py

├── .gitignore

└── README.md

```



Personal images, generated outputs, virtual environments, and model weights are excluded from Git.



\## ⚙️ Local Setup



\### Requirements



\- Windows 10/11 or a compatible operating system

\- Python 3.13 (the current development environment uses Python 3.13.13)

\- Git

\- Internet connection for initial dependency and model setup



\### 1. Clone the repository



```bash

git clone https://github.com/ajeem-suban/cullinan-ai.git

cd cullinan-ai

```



\### 2. Create a virtual environment



```bash

py -3.13 -m venv .venv

.venv\\Scripts\\activate

```



\### 3. Install dependencies



```bash

python -m pip install --upgrade pip

pip install insightface opencv-python numpy onnxruntime

```



\### 4. Prepare the face model



CULLINAN currently uses the `buffalo\_l` model pack from InsightFace.



The model must be available in the InsightFace model directory before running the face engine. InsightFace may download model files automatically during initialization. If you install the model manually, place the extracted model directory here:



```text

%USERPROFILE%\\.insightface\\models\\buffalo\_l\\

```



The model pack contains pretrained ONNX model files used by the face-analysis pipeline.



\### 5. Add your images



Place reference images in:



```text

photos/reference/

```



Supported image formats:



\- JPG

\- JPEG

\- PNG

\- WEBP



Use images that you own or have permission to process.



\## 🚀 Running CULLINAN



From the project root in Windows CMD:



```cmd

set "PYTHONPATH=%CD%\\src"

python test\_face.py

```



The program initializes the face engine, processes images in `photos/reference/`, and saves annotated images when faces are detected.



Generated images are saved in:



```text

output/

```



\## 🧪 Running Tests



Run the automated unit tests from the project root:



```bash

python -m unittest discover -s tests -v

```



The current tests validate:



\- Identical embedding similarity

\- Opposite embedding similarity

\- Identity-match threshold behavior

\- Different embedding comparisons

\- Embedding dimension validation

\- Zero-vector validation



These tests validate the comparison utilities. They do not establish real-world face recognition accuracy.



\## ⚡ Current Execution



The current configuration uses:



```python

EXECUTION\_PROVIDERS = ("CPUExecutionProvider",)

```



Inference currently runs on the CPU through ONNX Runtime.



GPU acceleration may be evaluated later after measuring actual performance and checking compatibility with the local hardware.



\## 🔐 Privacy



CULLINAN is designed around local image processing.



\- Personal photos are not included in the public repository.

\- Generated images are excluded from Git.

\- Local model weights are not included in Git.

\- Embeddings and databases should be treated as sensitive biometric data.

\- Only use images with appropriate permission.



\## ⚠️ Model Licensing



CULLINAN uses the InsightFace `buffalo\_l` pretrained model pack during development.



InsightFace's code license and pretrained model-weight terms are not necessarily the same. The `buffalo\_l` pretrained weights are subject to restrictions, including non-commercial research use.



Do not redistribute the pretrained weights or use them commercially without obtaining the appropriate permission or license. Review the official \[InsightFace repository](https://github.com/deepinsight/insightface) and its model licensing terms before deployment.



\## 🛣️ Roadmap



\- \[x] Establish the Python project structure

\- \[x] Set up the InsightFace environment

\- \[x] Initialize the face-analysis engine

\- \[x] Implement image-based face detection

\- \[x] Implement facial embedding comparison

\- \[x] Add initial automated tests

\- \[x] Publish the initial source code to GitHub

\- \[ ] Validate detection using personal reference images

\- \[ ] Build a complete identity-matching workflow

\- \[ ] Evaluate recognition accuracy and thresholds

\- \[ ] Benchmark CPU and GPU inference

\- \[ ] Explore personalized image generation

\- \[ ] Improve documentation and deployment



\## 👨‍💻 Developer



\*\*AJEEM S.\*\*



\- GitHub: \[@ajeem-suban](https://github.com/ajeem-suban)

\- Portfolio: \[ajeem-suban.vercel.app](https://ajeem-suban.vercel.app)

\- LinkedIn: \[ajeem-suban](https://www.linkedin.com/in/ajeem-suban/)



\---



\*\*CULLINAN — Building a personal AI system, one component at a time.\*\*



