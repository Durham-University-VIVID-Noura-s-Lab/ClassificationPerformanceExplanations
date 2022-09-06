# Classification Performance Explanations
---
This is the implementation of the paper "Generating Textual Explanations for Machine Learning Models Performance: A Table-to-Text Task" accepted for publication in LREC 2022, France. http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.379.pdf 

# Software Dependencies
---
The following are the main python packages used to in this work:
- pytorch
- transformers
- pytorch_lightning
- tqdm
- datasets

these can be installed by running the code  ``pip install -r requirement.txt`` 




### Running the codes
---

The models trained in this work are based on the T5 and BART pre-trained language models provided by huggingface. The variants of these models  supported are T5-small/base/large and BART-base/large.  
- The file ``trainer.py`` contains the source code for training the models to generate the textual narratives. The trained models as well as other outputs generated will be saved in the folder provided via the argument ``--output_dir``. 

- The file ``inference.py`` contains the source code for performing inference using the fine-tuned models. It requires the base-path (specified using the argument ``--model_base_dir``) to the  configuration files and the trained models generated after running the ``trainer.py`` file. 

- The file ``classification_performance_description_inference.py `` contains the code for generating the textual narratives for a given report.

To train a model and evaluate it run the code:

For simplicity, the locations for the training and testing datasets have been hard coded in the trainer.py file. Modify them to reflect the appropriate locations.

```
python trainer.py -run_id [any random id to identify the execution instance] \  
                  --modeltype [baseline | earlyfusion]  \
                  --modelbase [t5-small | t5-base | t5-large | facebook/bart-base | facebook/bart-large] \ 
                  --num_train_epochs 20 \
                  --learning_rate 3e-4 \
                  --output_dir [location for saving all the train model's configurations and files] \
```

Asumming the ``--output_dir`` is specified as ``narrative_models``, ``--modeltype`` as ``earlyfusion``, and ``--modelbase`` as  ``facebook/bart-base``, the structure of the ``--output_dir`` will be:
```
narrative_models
        /trainednarrators
           /earlyfusion
              /bart-base/
        
```
The files after training a model saved are:
```
parameters.json
pytorch_model.bin
```
* See the ``builder.ipynb`` for an example of the training routine


Sample of the inference routine is as follows:
* See the ``performance_narration.ipynb`` for the full code.

```
from classification_performance_description_inference import ClassificationPerformanceNarration

trained_model_path = 'TrainModels/earlyfusion/bart-base/'
#"Trained_models/trainednarrators/baseline/bart-base/"

modeltype='earlyfusion'
narrator = ClassificationPerformanceNarration(model_base = "facebook/bart-base",
                                              model_base_dir= trained_model_path,
                                              
                                              modeltype=modeltype,
                                              max_preamble_len=160,
                                              max_len_trg=185,
                                              length_penalty=8.6,
                                              beam_size=10,
                                              repetition_penalty=1.5,
                                              return_top_beams=1,
                                              random_state=453,
                                              )
                                              
# Set up the inference model
narrator.buildModel()

# The setup is ready to be used

# Sample of the expected format for the performance report
performance_report = {'Recall': ["19.56%", "LOW"],
                      'F1-score': ["9.06%",
                                   "LOW"],
                      'Accuracy': ["68.16%", "LOW"],
                      'AUC': ["50.16%", "MODERATE"]
                      }
class_labels = ['Not Fraud', 'Fraud']
is_balance = False
generatedTexts = narrator.generateTextualExplanation(performance_report,
                                                       class_labels,
                                                     is_balance=True)


print(generatedTexts)
```







