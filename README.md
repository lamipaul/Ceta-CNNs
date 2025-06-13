# Ceta CNNs
*Cetacean vocalization detection using pre-trained convolutionnal neural networks*

## Project description
This project allows to detect the vocalizations from several cetacean species in acoustic signals using Convolutionnal Neural Networks (CNNs).

CNN architectures with pretrained weights are available for inference through a python interface for the following species / signals:
- Humpback whale calls (*Megatera novaeangliae*, trained on recordings from the Caribbean Sea)
- Dolphin whistles (*Delphinid*, trained on recordings from the Caribbean Sea)
- Sperm whale clicks (*Physeter macrocephalus*, trained on recordings from the Mediterranean Sea)
- Fin whale 20Hz pulses (*Balaenoptera physalus*, trained on recordings from the Mediterranean Sea)
- Orcas pulsed calls (*Orcinus orca*, trained on recordings from the North-Eastern Pacific Ocean)

For the detection of Antarctic fin whale and and Antarctic blue whale vocalizations, see https://gitlab.lis-lab.fr/paul.best/at_bluefin

## Usage
Use python to run the script forward_CNN.py along with a target specie and a folder of audio files to analyse. A tabular file will be saved with the model's predictions for the corresponding signal to detect (probability of presence).

Run `python run_CNN.py -h` for a detailled API.

The output file with predictions can be read in python using pandas : `pandas.read_pickle('filename.pkl')`

## Dependencies
This script relies on `torch`, `pandas`, `numpy`, `scipy`, `soundfile` and `tqdm` to run. You can install them using pip or conda.
If a GPU and cuda are available on the current machine, processes will run on GPU for faster computation.

## How to cite
APA:
```
Best, P. (2022). Ceta-CNNs: Cetacean vocalization detection using pre-trained convolutionnal neural networks. GitHub, https://github.com/lamipaul/Ceta-CNNs.
```
Bibtex:
```
@misc{best_2022,
    author       = {Paul Best},
    title        = {{Ceta-CNNs: Cetacean vocalization detection using pre-trained convolutionnal neural networks}},
    year         = 2022,
    version      = {1.0},
    url          = {https://github.com/lamipaul/Ceta-CNNs}
    }
```

## Contact
You can reach me at paul.best@univ-amu.fr for more information

