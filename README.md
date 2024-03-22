# References
When using this code and/or the LISI-HHI dataset, please cite the following works:

#### A multimodal dataset for robot learning to imitate social human-human interaction
```
@inproceedings{tuyen2023multimodal,
  title={A multimodal dataset for robot learning to imitate social human-human interaction},
  author={Tuyen, Nguyen Tan Viet and Georgescu, Alexandra L and Di Giulio, Irene and Celiktutan, Oya},
  booktitle={Companion of the 2023 ACM/IEEE International Conference on Human-Robot Interaction},
  pages={238--242},
  year={2023}
}
```

#### LISI-HHI: A Dataset for Studying Human Non-verbal Signals in Dyadic Interactions
```
TBA
```

# Setup
To set up the training and evaluation, you need the LISI dataset's pickle files downloaded. You can ask the authors for the zipped version of this. These files need to be extracted into the `PreProcessData` folder.
In addition, we are also happy to provide a zip with the trained model's checkpoint upon request. The checkpoint folder needs to be extracted into the `fullmodels/model_SessALL` folder.

To run the scripts, set up a virtual environment - conda recommended. Then install *Tensorflow 1.14* - if you are using conda, after activating the environment, you can do so with 
```
conda install -c conda-forge tensorflow=1.14
```

Then, with the environment still active, install the required packages from the `requirements.txt`. You can do so with:
```
pip install -r requirements.txt
```

# Usage
To train the model:
```
cd fullmodels
python3 train-SessALL.py 
```

To evaluate the model:
```
cd fullmodels
python3 evaluate_SessALL.py 
```

To run visualisation only in a jupyter notebook:
```
cd <ROOT OF REPO>
jupyter notebook
```
Then in the `fullmodels` folder open `Visualize-generated.ipynb`. Set up your paths, but most of them should be relative, and run all the cells.
