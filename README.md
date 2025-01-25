# contrastive-da-synthetic-patch

## Repository Structure

<pre>
data/
    └── synthetic_patches/
        └── images
        └── datalist_train_valid.json  # Datalist containing image path and labels
        └── datalist_test.json  # Datalist containing image path and labels
        └── specs.json  # Parameters used for patch generation
        └── annots.csv  # Annotations
    └── synthesize_mammo_patches.py # Source code for patch generation
src/ # Source code for training the models
third_party/ # Third party code used (Supervised Contrastive loss computation)
train_contrastive.py  # Script for training a contrastive model
train_cross_entropy.py  # Script for training a cross-entropy model
.gitignore  # Specifies which files should be ignored by git
.gitattributes  # Git attributes file for handling line endings and other settings
README.md  # This file
</pre>

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

Additional arguments can be included when launching the scripts:

```
python src/train_cross_contrastive.py --batch_size 128 --learning_rate_contr 0.01
```
