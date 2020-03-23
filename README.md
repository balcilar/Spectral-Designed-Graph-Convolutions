# Spectral Designed Graph Convolutions

Codes of ["Bridging the Gap Between Spectral and Spatial Domains in Graph Neural Networks"](https://www.google.com) paper.


## Requirements
These libraries versions are not stricly needed. But these are the configurations in our test machine.
- Python==3.6.5
- tensorflow-gpu==1.15.0
- numpy==1.17.4
- networkx==2.4
- scipy==1.3.1
- matplotlib==3.1.2
- pickle==4.0

## Usage
Run the scripts directly. All parameters are defined in corresponding script. In Pubmed and PPI dataset, since the eigen decomposition takes quite a long time because of the dimension of given graph, we write the eigenvectors into file in first run. For later run, the code directly read already calculated eigenvectors from file.

### Transductive Setting Problems
	python cora_multirun.py
	python citeseer_multirun.py
	python pubmed_multirun.py
### Inductive Setting Problems
	python ppi_singlerun.py
	python protein_nodelabel.py
	python enzymes_nodelabel.py
	python enzymes_allfeats.py

## Results

![Sample image](logs/result.jpg?raw=true "Title")

## Citation

Please cite this paper if you want to use it in your work,

	@article{spatialspectralgap,
	  title={Bridging the Gap Between Spectral and Spatial Domains in Graph Neural Networks},
	  author={Balcilar,Muhammet  and Renton, Guillaume and H\'eroux,Pierre and Ga\"uz\`ere,Benoit and Adam, S\'ebastien and Honeine,Paul},
	  journal={arXiv preprint arXiv:xxxx.xxxxx},
	  year={2020}
	}

  
## License
MIT License
