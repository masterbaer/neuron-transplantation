# Neuron Transplantation
This is the repository for the paper **Model Fusion via Neuron Transplantation**.  

![transplantation drawio](https://github.com/masterbaer/ensemble-fusion/assets/56799329/034ecee8-a311-4360-a8b8-371f0d6be449)


# Method  

Neuron Transplantation fuses multiple fully trained ensemble members by transplanting the largest neurons from each model.  
In order to do this, we first concatenate all the ensemble members into a large model and then prune it using torch-pruning.  
The method is illustrated in the following image:  

![concat_pruning_process drawio](https://github.com/masterbaer/ensemble-fusion/assets/56799329/0145e78c-de51-4730-ba3a-bc5f4f991ed7)

This process sacrifices the smaller neurons to make space for better, larger neurons from the other ensemble members.  
The following image shows two direct consequences:  

![ex9](https://github.com/masterbaer/ensemble-fusion/assets/56799329/8c134400-68b8-402b-b553-14d6b082e6e5)  

Firstly, the sacrifice of the smaller neurons causes some initial damage leading to some post-fusion loss.  
Secondly, the newly transplanted neurons lead to better performance after fine-tuning.  

Note: If the newly transplanted neurons are redundant (from the same model or from very similar models) then the transplantation (obviously) does not improve the 
performance.

# Repository Overview 
In "fusion_methods" the model fusion methods are implemented. Among them are Vanilla Averaging, Optimal Transport Fusion (from Singh and Jaggi) and Neuron Transplantation (ours).
Feel free to simply copy the file neuron_transplantation.py to try it out.  

The folder "experiments" contains several sanity checks and experiments.  
In the paper, the experiments 6, 7a-d, 8a-b, 9 and 11 and 12 are used.   

# Example usage
```python
from fusion_methods.neuron_transplantation import fuse_ensemble  

... # train models  

models = [model0, model1] # fully trained models, 2 or more  

for images, labels in train_loader:  
  example_inputs = images # torch-pruning requires some example inputs to work  
  break  
  
fused_model = fuse_ensemble(models, example_inputs)  
```



# Running the experiments
1) Load the necessary modules:
```   
module purge  
module load compiler/gnu/11  
module load devel/cuda/11.8    
module load mpi/openmpi/4.0  
module load lib/hdf5/1.12  
```
2) Create a virtual environment:  
```
python -m venv ~/venv/neurontransplantation
```
3) Activate virtual environment:  
```
source venv/bin/activate
```
4) Install requirements:
```
pip install -r requirements.txt
```
If the cuda version is not correct automatically, then install torch manually (e.g. on windows/linux) :  
```
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118.  
```

5) Submit experiments e.g. using  
```
sbatch experiments/ex6_submit.sh  
```
or run the python files directly with their command line arguments using  
```
python ex6_order.py "p_m_ft" 0
```

# Cite
TODO

## Acknowledgements

*This work is supported by the [Helmholtz Association Initiative and
Networking Fund](https://www.helmholtz.de/en/about_us/the_association/initiating_and_networking/)
under the Helmholtz AI platform grant and the HAICORE@KIT partition.*

---

<div align="center">
    <a href="https://www.helmholtz.ai/"><img src="logos/helmholtzai_logo.jpg" height="45px" hspace="3%" vspace="20px"></a><a href="http://www.kit.edu/english/index.php"><img src="logos/kit_logo.svg" height="45px" hspace="3%" vspace="20px"></a><a 
</div>
