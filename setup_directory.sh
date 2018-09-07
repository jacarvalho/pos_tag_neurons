# Run this script to install the required packages in a virtual environment
virtualenv -p python3 venv
source venv/bin/activate
pip3 install --upgrade setuptools pip
pip3 install -r requirements.txt
python3 setup.py install
deactivate

# Concatenate trained lstm models together
cat byte_LSTM_trained_models/amazon/save/806/model-1927169.data-00000-of-00001.a* > byte_LSTM_trained_models/amazon/save/806/model-1927169.data-00000-of-00001
cat byte_LSTM_trained_models/wikitext/save/95/model-161909.data-00000-of-00001.a* > byte_LSTM_trained_models/wikitext/save/95/model-161909.data-00000-of-00001
