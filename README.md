# contrastive-da-synthetic-patch

## Repository Structure

data/
    └── synthetic_patches/
        └── datalist_train_valid.json  # Synthetic dataset used for training and validation
src/
train_contrastive.py  # Script for training a contrastive model
train_cross_entropy.py  # Script for training a cross-entropy model
.gitignore  # Specifies which files should be ignored by git
.gitattributes  # Git attributes file for handling line endings and other settings
README.md  # This file

## Setup and Installation

### Dependencies

To get started with the project, you need to install the required Python dependencies. It's recommended to create a virtual environment for this project:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
pip install -r requirements.txt
```

## Training the models
```
python src/train_cross_entropy.py
```
```
python src/train_cross_contrastive.py
```
