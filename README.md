# RaKE

This is the repository for the paper [Assessing Knowledge Editing in Language Models via Relation Perspective](https://arxiv.org/abs/2311.09053)

In real-world scenarios, such as Wikipedia, updating factual knowledge sometimes necessitates the modification of relationships to accurately reflect evolving information.
Consequently, this paper proposes an editing problem variants (**Relation-based Editing**), and provides a new benchmark named **RaKE**.

![example](https://github.com/weiyifan1023/RaKE/blob/main/example.jpg){:width="100px"}


The experimental results reveal that relation-based editing lags far behind entity-based editing, even though they should ideally be consistent since the original and the altered triples are the same. 

![variants](https://github.com/weiyifan1023/RaKE/blob/main/editing_problem_variants.png)

We currently support OpenAI's GPT-2 XL (1.5B) and EleutherAI's GPT-J (6B).  We hope that our work can provide the NLP community with insights



## Requirements

- At least one A6000 48G GPU and another GPU with no less than 24G memory.

- Environment

  ```
  conda create -n RaKE python=3.9.7
  pip install -r requirements.txt
  ```


## Baselines

The current supported knowledge editing techniques are as follows:

- [FT](https://github.com/kmeng01/rome): Fine-Tuning 
- [KN](https://github.com/Hunter-DDM/knowledge-neurons): Damai Dai et al. Locate then Edit
- [MEND](https://github.com/eric-mitchell/mend): Mitchell et al. Memory-based
- [ROME](https://github.com/kmeng01/rome): Kevin Meng et al. Locate and Edit
- [MEMIT](https://github.com/kmeng01/memit): Kevin Meng et al. Locate and Edit

We will support In-Context Editing ([IKE](https://github.com/Zce1112zslx/IKE)) soon.

## Edit Language Models

See [`baselines/`](baselines/) for a description of the available baselines.

### Running the Editor methods

[`experiments/evaluate.py`](experiments/evaluate.py) can be used to evaluate any method in [`baselines/`](baselines/).
To get started (e.g. selecting Editor such as ROME on GPT-2 XL), run via entity perspective:

```bash
python3 -m experiments.evaluate \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --perspective=entity
```

To get started (e.g. selecting Editor such as MEND on GPT-J 6B ), run via relation perspective:

```bash
python3 -m experiments.evaluate \
    --alg_name=MEND \
    --model_name=gpt-j-6b \
    --hparams_fname=EleutherAI_gpt-j-6B.json \
    --perspective=relation
```



Results from each run are stored at `results/<editor_name>/run_<run_id>` in a specific format:

```bash
results/
|__ <editor_name>/
    |__ run_<run_id>/
        |__ params.json
        |__ case_0.json
        |__ case_1.json
        |__ ...
        |__ case_7500.json
```

### Assessing the Editing Results

To summarize the results, you can use [`experiments/summarize.py`](experiments/summarize.py):

```bash
python3 -m experiments.summarize --dir_name=<editor_name> --runs=run_<run_id>
```

Running `python3 -m experiments.evaluate -h` or `python3 -m experiments.summarize -h` provides details about command-line flags.

Finally, run the main scripts:
```bash
python3 -m experiments.evaluate \
    --alg_name=<editor_name> \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json

python3 -m experiments.summarize --dir_name=X --runs=run_<run_id>
```

## Acknowledgment

Our code is based on [ROME](https://github.com/kmeng01/rome) and [EasyEdit](https://github.com/zjunlp/EasyEdit.git).

## Citation
If you use this code for your research, please kindly cite our paper:

```bibtex
@misc{wei2023assessing,
      title={Assessing Knowledge Editing in Language Models via Relation Perspective}, 
      author={Yifan Wei and Xiaoyan Yu and Huanhuan Ma and Fangyu Lei and Yixuan Weng and Ran Song and Kang Liu},
      year={2023},
      eprint={2311.09053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact

Yifan Wei: weiyifan2021@ia.ac.cn (Preferred)  &&  weiyifan21@mails.ucas.ac.cn 

