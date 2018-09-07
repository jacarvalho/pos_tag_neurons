help:
	@echo "\nFor a full description of the directories and files contained \
in this project, and how to run it, please read the README.txt file.\n"
	@echo "To train the LSTM language models, we don't recommend running it in \
docker, but use instead the riseml platform.\nAlso for the training of \
logistic regression classifiers, it is best to use a machine with more RAM \
(16-32 GB should be enough)."
	@echo "More information can be found in the README files in this folder \
or in the corresponding subfolders.\n"
	@echo "\nWith this makefile you can start a webapp that showcases the \
work done during this project by running:\n"
	@echo "make start_webapp"
	@echo "  - starts the webapp in localhost at port 5000."
	@echo "  - files read: trained LSTM models in \
/nfs/students/joao-carvalho/byte_LSTM_trained_models; html, css, js files in \
/www directory"
	@echo "  - files produced: none"
	@echo "  - RAM used: ~700MB\n" 
	

start_webapp:
	cd /www && python3 /www/app.py	
