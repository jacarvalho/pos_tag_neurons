# Run this script to install the python virtualenv and the required libraries
virtualenv -p python3 venv
source venv/bin/activate
pip3 install --upgrade setuptools pip
pip3 install -r requirements.txt
python3 -m nltk.downloader treebank
python3 -m nltk.downloader maxent_treebank_pos_tagger
