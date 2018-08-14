help:
	@echo "\nWith this makefile you can start a webapp that displays all the \
work done during this project.\n"
	@echo "make start_webapp: starts the webapp in localhost at port 5000."
	@echo "  - files read: trained LSTM models in \
/nfs/students/joao-carvalho/byte_LSTM_trained_models"
	@echo "  - memory usage: ~50MB of RAM; irrelevant disk space."
	@echo "\n"
	@echo "To train the LSTM language models, we don't recommend running in \
docker, but use instead the riseml platform.\nAlso for the training of \
logistic regression classifiers, it is best to use a machine with more RAM \
(16-32 GB should be enough)."
	@echo "More information can be found in the README file in this folder \
or in the corresponding subfolders.\n"
	

start_webapp:
	cd /code/www && python3 /code/www/app.py	
