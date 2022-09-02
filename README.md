# Classification Performance Explanations
This is the implementation of the paper "Generating Textual Explanations for Machine Learning Models Performance: A Table-to-Text Task" accepted for publication in LREC 2022, France. http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.379.pdf 

# Software Dependencies
The following are the main python packages used to in this work:
- pytorch
- transformers
- pytorch_lightning
- tqdm
- datasets

these can be installed by running the code  ``pip install -r requirement.txt`` 

# The project
The models trained in this work are based on the T5 and BART pre-trained language models provided by huggingface. The variants of these models  supported are T5-small/base/large and BART-base/large.  
- The file ``trainer.py`` contains the source code for training the models to generate the textual narratives. The trained models as well as other outputs generated will be saved in the folder provided via the argument ``--output_dir``. 
 
