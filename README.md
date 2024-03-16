# dyMEAN: End-to-End Full-Atom Antibody Design

This repo contains the codes for our paper [End-to-End Full-Atom Antibody Design](https://arxiv.org/abs/2302.00203).


## Quick Links

- [Setup](#setup)
- [Experiments](#experiments)
    - [Data Preprocessing](#data-preprocessing)
    - [CDR-H3 Design](#cdr-h3-design)
    - [Complex Structure Prediction](#complex-structure-prediction)
    - [Affinity Optimization](#affinity-optimization)
- [Proof-of-Concept Applications](#proof-of-concept-applications)
    - [Inference API](#inference-api)
    - [*In Silico* "Display"](#in-silico-display)
- [Contact](#contact)
- [Others](#others)


## Setup

There are 3 necessary and 1 optional prerequisites: setting up conda environment (necessary), obtaining scorers (necessary), preparing antibody pdb data (necessary), and downloading baselines (optional).

**1. Environment**

We have provided the `env.yml` for creating the runtime conda environment just by running:

```bash
conda env create -f env.yml
```

**2. Scorers**

Please first prepare the scorers for TMscore and DockQ as follows:

The source code for assessing TMscore is at `evaluation/TMscore.cpp`. Please compile it by:
```bash
g++ -static -O3 -ffast-math -lm -o evaluation/TMscore evaluation/TMscore.cpp
```

To prepare the DockQ scorer, please clone its [official github](https://github.com/bjornwallner/DockQ) and compile the prerequisites according to its instructions. After that, please revise the `DOCKQ_DIR` variable in the `configs.py` to point to the directory containing the DockQ project (e.g. ./DockQ).

The lDDT scorer is in the conda environment, and the $\Delta\Delta G$ scorer is integrated into our codes, therefore they don't need additional preparations.

**3. PDB data**

Please download all the structure data of antibodies from the [download page of SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?all=true). Please enter the *Downloads* tab on the left of the web page and download the archived zip file for the structures, then decompress it:

```bash
wget https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/ -O all_structures.zip
unzip all_structures.zip
```

You should get a folder named *all_structures* with the following hierarchy:

```
├── all_structures
│   ├── chothia
│   ├── imgt
│   ├── raw
```

Each subfolder contains the pdb files renumbered with the corresponding scheme. We use IMGT in the paper, so the imgt subfolder is what we care about.

Since pdb files are heavy to process, usually people will generate a summary file for the structural database which records the basic information about each structure for fast access. We have provided the summary of the dataset retrieved at November 12, 2022 (`summaries/sabdab_summary.tsv`). Since the dataset is updated on a weekly basis, if you want to use the latest version, please download it from the [official website](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/about/).


**(Optional) 4. Baselines**

If you are interested in the pipeline baselines, including the following projects and integrate their dependencies according to your needs:

- framework structure prediction:
    - [IgFold](https://github.com/Graylab/IgFold/tree/main/igfold)
- docking:
    - [HDock](http://huanglab.phys.hust.edu.cn/software/hdocklite/)
- CDR design:
    - [MEAN](https://github.com/THUNLP-MT/MEAN)
    - [Diffab](https://github.com/luost26/diffab)
    - [Rosetta](https://new.rosettacommons.org/demos/latest/tutorials/install_build/install_build)
- side-chain packing:
    - [Rosetta](https://new.rosettacommons.org/demos/latest/tutorials/install_build/install_build)

After adding these projects, please also remember to revise the corresponding paths in `./configs.py`. We have also provided the scripts for cascading the modules in `./scripts/pipeline_inference.sh`.


## Experiments

The trained checkpoints for each task are provided at the [github release page](https://github.com/THUNLP-MT/dyMEAN/releases/tag/v1.0.0). To use them, please download the ones you are interested in and save them into the folder `./checkpoints`. We provide the names, training configurations (under `./scripts/train/configs`), and descriptions of the checkpoints as follows:

| checkpoint(s)                                | configure              | description                                    |
| -------------------------------------------- | ---------------------- | ---------------------------------------------- |
| cdrh3_design.ckpt                            | single_cdr_design.json | Epitope-binding CDR-H3 design                  |
| struct_prediction.ckpt                       | struct_prediction.json | Complex structure prediction                   |
| affinity_opt.ckpt & ddg_predictor.ckp        | single_cdr_opt.json    | Affinity optimization on CDR-H3                |
| multi_cdr_design.ckpt                        | multi_cdr_design.json  | Design all 6 CDRs simultaneously               |
| multi_cdr_opt.ckpt & multi_cdr_ddg_predictor | multi_cdr_opt.json     | Optimize affinity on all 6 CDRs simultaneously |
| full_design.ckpt                             | full_design.json       | Design the entire variable domain, including the framework region |

### Data Preprocessing

**Data**

To preprocess the raw data, we need to first generate summaries for each benchmark in json format, then split the datasets into train/validation/test sets, and finally transform the pdb data to python objects. We have provided the script for all these procedures in `scripts/data_preprocess.sh`. Suppose the IMGT-renumbered pdb data are located at `all_structures/imgt/`, and that you want to store the processed data (~5G) at `all_data`, you can simply run:

```bash
bash scripts/data_preprocess.sh all_structures/imgt all_data
```
which takes about 1 hour to process SAbDab, RAbD, Igfold test set, and SKEMPI V2.0. It is normal to see reported errors in this process because some antibody structures are wrongly annotated or have wrong format, which will be dropped out in the data cleaning phase.

**(Optional) Conserved Template**

We have provided the conserved template from SAbDab in `./data/template.json`. If you are interested in the extracting process, it is also possible to extract a conserved template from a specified dataset (e.g. the training set for the CDR-H3 design task) by running the following command:

```bash
python -m data.framework_templates \
    --dataset ./all_data/RAbD/train.json \
    --out ./data/new_template.json
```


### CDR-H3 Design
We use SAbDab for training and RAbD for testing. Please first revise the settings in `scripts/train/configs/cdr_design.json` (path to datasets and other hyperparameters) and then run the below command for training:
```bash
GPU=0,1 bash scripts/train/train.sh scripts/train/configs/single_cdr_design.json
```
Normally the training procedure takes about 7 hours on 2 GeForce RTX 2080 Ti GPUs. We have also provided the trained checkpoint at `checkpoints/cdrh3_design.ckpt`. Then please revise the path to the test set in `scripts/test/test.sh` and run the following command for testing:
```bash
GPU=0 bash scripts/test/test.sh ./checkpoints/cdrh3_design.ckpt ./all_data/RAbD/test.json ./results
```
which will save the generated results to `./results`.

### Complex Structure Prediction
We use SAbDab for training and IgFold for testing. The training and testing procedure are similar to those of CDR-H3 design. After revising the settings in `scripts/train/configs/cdr_design.json` and `scripts/test/test.sh` as mentioned before, please run the following command for training:

```bash
GPU=0,1 bash scripts/train/train.sh scripts/train/configs/struct_prediction.json
```
Normally the training procedure takes about 8 hours on 2 GeForce RTX 2080 Ti GPUs. We have also provided the trained checkpoint at `checkpoints/struct_prediction.ckpt`. Then please run the following command for testing:
```bash
GPU=0 bash scripts/test/test.sh ./checkpoints/struct_prediction.ckpt ./all_data/IgFold/test.json ./results
```

### Affinity Optimization
We use SAbDab for training and the antibodies in SKEMPI V2.0 for testing. Similarly, please first revise the settings in `scripts/train/configs/affinity_opt.json`, `scripts/test/optimize_test.sh`, and additionally `scripts/train/train_predictor.sh`. Then please conduct training of dyMEANOpt (~ 5h):
```bash
GPU=0,1 bash scripts/train/train.sh scripts/train/configs/single_cdr_opt.json
```
Then we need to train a predictor of ddg on the representations of generated complex (~ 40min):
```bash
GPU=0 bash scripts/train/train_predictor.sh checkpoints/cdrh3_opt.ckpt
```
We have provided the trained checkpoints at `checkpoints/cdrh3_opt.ckpt` and `checkpoints/cdrh3_ddg_predictor.ckpt`. The optimization test can be conducted through:
```bash
GPU=0 bash scripts/test/optimize_test.sh checkpoints/cdrh3_opt.ckpt checkpoints/cdrh3_ddg_predictor.ckpt ./all_data/SKEMPI/test.json 0 50
```
which will do 50 steps of gradient search without restrictions on the maximum number of changed residues (change 0 to any number to restrict the upperbound of $\Delta L$).


## Proof-of-Concept Applications

We also provide inference API and *in silico* demos for common applications in the real world problems, which are located in the `./api` and `./demos`.

### Inference API

We provide the **design** API and the **optimize** API in `./api`, which can be easily integrated into python codes.

#### Design

The **design** API (`./api/design.py`) can be used to generate CDRs given the sequences of the framework region, the PDB file of the antigen as well as the epitope definitions. We will use the an interesting scenario to illustrate the usage of the **design** API.

We want to design an antibody combining to the open state of the transient receptor potential cation channel subfamily V member 1 (TRPV1), which plays a critical role in acute and persistent pain. Instead of handcraft the epitope on TRPV1, we try to mimic an existing binder which is a double-knot toxin (DkTx). Therefore, we need to first extract the epitope definition by analyzing the binding pattern of the toxin, then design an antibody with given sequences of the framework regions.

**1. Extract the Epitope Definition**

We provide the PDB file of the complex of the transient receptor potential cation channel subfamily V member 1 (TRPV1, chain ABCD) and the double-knot toxin (DkTx, chain EF) in `./demos/data/7l2m.pdb`. The original PDB has 4 symmetric units, so we manually split the two toxins (chain EF) in the middle to form 4 symmetric chains e,f,E,F. Each antibody only need to focus on one unit. Here we choose the chain E as an example.

We generate the epitope definition by analyzing the binding interface of chain E to the TRPV1:

```bash
python -m api.binding_interface \
    --pdb ./demos/data/7l2m.pdb \
    --receptor A B C D \
    --ligand E \
    --out ./demos/data/E_epitope.json
```

Now the epitope definition (i.e. the residues of TRPV1 on the binding interface) is saved to `./demos/data/E_epitope.json`. By changing the value of the argument "ligand" to e, f, and F, we can obtain the epitope definitions for other units (don't forget to revise the output path as well).

**2. Obtain the Sequences of the Framework Regions**

Depending on the final purposes of designing the antibody, framework regions with different physiochemical properties may be desired. Since here we are only providing a proof-of-concept case, we randomly pick up one from the existing dataset:

```bash
heavy chain (H): 'QVQLKESGPGLLQPSQTLSLTCTVSGISLSDYGVHWVRQAPGKGLEWMGIIGHAGGTDYNSNLKSRVSISRDTSKSQVFLKLNSLQQEDTAMYFC----------WGQGIQVTVSSA'
light chain (L): 'YTLTQPPLVSVALGQKATITCSGDKLSDVYVHWYQQKAGQAPVLVIYEDNRRPSGIPDHFSGSNSGNMATLTISKAQAGDEADYYCQSWDGTNSAWVFGSGTKVTVLGQ'
```

The original CDR-H3 is masked by "-". Designing multiple CDRs are also supported, which will be illustrated later.

**3. Design the CDRs**

The last step is to design the CDRs with the **design** API:

```python
from api.design import design

ckpt = './checkpoints/cdrh3_design.ckpt'
root_dir = './demos/data'
pdbs = [os.path.join(root_dir, '7l2m.pdb') for _ in range(4)]
toxin_chains = ['E', 'e', 'F', 'f']
remove_chains = [toxin_chains for _ in range(4)]
epitope_defs = [os.path.join(root_dir, c + '_epitope.json') for c in toxin_chains]
identifiers = [f'{c}_antibody' for c in toxin_chains]

# use '-' for masking amino acids
frameworks = [
    (
        ('H', 'QVQLKESGPGLLQPSQTLSLTCTVSGISLSDYGVHWVRQAPGKGLEWMGIIGHAGGTDYNSNLKSRVSISRDTSKSQVFLKLNSLQQEDTAMYFC----------WGQGIQVTVSSA'),
        ('L', 'YTLTQPPLVSVALGQKATITCSGDKLSDVYVHWYQQKAGQAPVLVIYEDNRRPSGIPDHFSGSNSGNMATLTISKAQAGDEADYYCQSWDGTNSAWVFGSGTKVTVLGQ')
    ) \
    for _ in pdbs
]  # the first item of each tuple is heavy chain, the second is light chain

design(ckpt=ckpt,  # path to the checkpoint of the trained model
       gpu=0,      # the ID of the GPU to use
       pdbs=pdbs,  # paths to the PDB file of each antigen (here antigen is all TRPV1)
       epitope_defs=epitope_defs,  # paths to the epitope definitions
       frameworks=frameworks,      # the given sequences of the framework regions
       out_dir=root_dir,           # output directory
       identifiers=identifiers,    # name of each output antibody
       remove_chains=remove_chains,# remove the original ligand
       enable_openmm_relax=True,   # use openmm to relax the generated structure
       auto_detect_cdrs=False)  # manually use '-'  to represent CDR residues
```

These codes are also added as an example in `./api/design.py`, so you can directly run it by:

```bash
python -m api.design
```

Here we use "-" to mark the CDR-H3 manually, but you can also set `auto_detect_cdrs=True` to let the CDR be automatically decided by the IMGT numbering system. The types of the CDRs to design will be automatically derived from the given checkpoint. Currently the API support re-designing single or multiple CDRs, as well as designing the full antibody (by passing `"-" * length` as the input).

Enabling Openmm relax will slow down the generation process a lot, but will rectify the bond lengths and angles to conform to the physical constraints.

#### Optimize

The **optimize** API (`./api/optimize.py`) is straight-forward. We optimize `./demos/data/1nca.pdb` as an example:

```python

from api.optimize import optimize, ComplexSummary

ckpt = './checkpoints/cdrh3_opt.ckpt'
predictor_ckpt = './checkpoints/cdrh3_ddg_predictor.ckpt'
root_dir = './demos/data/1nca_opt'
summary = ComplexSummary(
    pdb='./demos/data/1nca.pdb',
    heavy_chain='H',
    light_chain='L',
    antigen_chains=['N']
)
optimize(
    ckpt=ckpt,  # path to the checkpoint of the trained model
    predictor_ckpt=predictor_ckpt,  # path to the checkpoint of the trained ddG predictor
    gpu=0,      # the ID of the GPU to use
    cplx_summary=summary,   # summary of the complex as well as its PDB file
    num_residue_changes=[1, 2, 3, 4, 5],  # generate 5 samples, changing at most 1, 2, 3, 4, and 5 residues, respectively
    out_dir=root_dir,  # output directory
    batch_size=16,     # batch size
    num_workers=4,     # number of workers to use
    optimize_steps=50  # number of steps for gradient desend
)
```

Codes for this example is also added to `./api/optimize.py`, so you can directly run them by:

```bash
python -m api.optimize
```

Then you will get the following results:
```
├── demos/data/1nca_opt
│   ├── 1nca_0_1.pdb
│   ├── 1nca_1_2.pdb
│   ├── 1nca_2_3.pdb
│   ├── 1nca_3_4.pdb
│   ├── 1nca_4_5.pdb
│   ├── 1nca_original.pdb
```
where the `1nca_original.pdb` is the original complex, and `1nca_a_b.pdb` means the $a$-th candiates with constraints of changing up to $b$ residues.

#### Complex Structure Prediction

The **complex structure prediction** API (`./api/structure_prediction.py`) predicts the complex structure given the antigen, the sequences of the heavy chain and the light chain, and the definition of the epitope. Global docking is still very challenging so we narrow the scope to the epitope of interest. We predict`./demos/data/1nca.pdb` as an example:

```python

from api.structure_prediction import structure_prediction

ckpt = './checkpoints/struct_prediction.ckpt'
root_dir = './demos/data'
n_sample = 10  # sample 10 conformations
pdbs = [os.path.join(root_dir, '1nca_antigen.pdb') for _ in range(n_sample)]
epitope_defs = [os.path.join(root_dir, '1nca_epitope.json') for _ in range(n_sample)]
identifiers = [f'1nca_model_{i}' for i in range(n_sample)]

seqs = [
    (
        ('H', 'QIQLVQSGPELKKPGETVKISCKASGYTFTNYGMNWVKQAPGKGLKWMGWINTNTGEPTYGEEFKGRFAFSLETSASTANLQINNLKNEDTATFFCARGEDNFGSLSDYWGQGTTVTVSS'),
        ('L', 'DIVMTQSPKFMSTSVGDRVTITCKASQDVSTAVVWYQQKPGQSPKLLIYWASTRHIGVPDRFAGSGSGTDYTLTISSVQAEDLALYYCQQHYSPPWTFGGGTKLEIK')
    ) \
    for _ in pdbs
]  # the first item of each tuple is heavy chain, the second is light chain

structure_prediction(
    ckpt=ckpt,  # path to the checkpoint of the trained model
    gpu=0,      # the ID of the GPU to use
    pdbs=pdbs,  # paths to the PDB file of each antigen (here antigen is all TRPV1)
    epitope_defs=epitope_defs,  # paths to the epitope definitions
    seqs=seqs,      # the given sequences of the framework regions
    out_dir=root_dir,           # output directory
    identifiers=identifiers,    # name of each output antibody
    enable_openmm_relax=True)   # use openmm to relax the generated structure

```

Codes for this example is also added to `./api/structure_prediction.py`, so you can directly run them by:

```bash
python -m api.structure_prediction
```

Then you will get the following results:
```
├── demos/data
│   ├── 1nca_model_0.pdb
│   ├── 1nca_model_1.pdb
│   ├── 1nca_model_2.pdb
│   ├── ...
```
where there should be a total of 10 sampled conformations. Note that the first or last few residues might be discarded in the results if they are out of the variable domain according to the IMGT numbering system.


### *In Silico* "Display"

*In vitro* display are commonly used for selecting binding mutants from antibody libraries. Here we implement an *in silico* version with the **design** API by generating and filtering candidates from existing dataset against the antigen with an epitope definition. Further, we need an metric to evaluate how well the generated antibody binds to the target. Here we use FoldX as the affinity predictor, so to run this demo, you may need to first download the it from the [official website](https://foldxsuite.crg.eu/products#foldx) and revise the path in `./configs.py` correspondingly. We still use the TRPV1 example in the previous section, and use the RAbD benchmark as the antibody library providing the framework regions:

```bash
python -m demos.display \
    --ckpt checkpoints/multi_cdr_design.ckpt \
    --pdb demos/data/7l2m.pdb \
    --epitope_def demos/data/E_epitope.json \
    --library ./all_data/rabd_all.json \
    --n_sample 30 \
    --save_dir demos/display \
    --gpu 0
```

which will results in 30 candidates with their affinity predicted by FoldX.


## Contact

Thank you for your interest in our work!

Please feel free to ask about any questions about the algorithms, codes, as well as problems encountered in running them so that we can make it clearer and better. You can either create an issue in the github repo or contact us at jackie_kxz@outlook.com.

## Others
The files below are borrowed from existing repositories:

- `evaluation/TMscore.cpp`: https://zhanggroup.org/TM-score/
- `evaluation/ddg`: https://github.com/HeliXonProtein/binding-ddg-predictor
