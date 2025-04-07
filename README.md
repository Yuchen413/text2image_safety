# SneakyPrompt: Jailbreaking Text-to-image Generative Models

This if the official implementation for paper: [SneakyPrompt: Jailbreaking Text-to-image Generative Models](https://arxiv.org/abs/2305.12082)

Our work has been reported by [MIT Technology Review](https://www.technologyreview.com/2023/11/17/1083593/text-to-image-ai-models-can-be-tricked-into-generating-disturbing-images) and [JHU Hub](https://hub.jhu.edu/2023/11/01/nsfw-ai/). Please check them out if interested.


## Environment setup

The experiment is run on Ubuntu 18.04, with one Nvidia 3090 GPU (24G). Please install the dependencies via:

``conda env create -f environment.yml``

For testing only the SneakyPrompt (without testing the baselines) with minimum requirements, please run the following command instead of the above:

``conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia``

``pip install transformers==4.27.4 accelerate==0.18.0 sentencepiece==0.1.97 einops==0.7.0 triton==2.1.0 diffusers==0.29.2 numpy==1.26.0 xformers==0.0.22.post7 tensorflow==2.8.3 pandas pillow scikit-learn protobuf torchmetrics matplotlib``

``pip install git+https://github.com/openai/CLIP.git``


## Dataset

The [nsfw_200.txt](https://livejohnshopkins-my.sharepoint.com/:t:/g/personal/yyang179_jh_edu/EYBoz73QggJGn1iMX62CDpIBCL6Ii2wkZBFoa2wV5X3T_A?e=9G8nar) can be accessed per request. To ensure responsible use, please fill the [request form](https://docs.google.com/forms/d/e/1FAIpQLSdVrav-vi0NcbLuha3t-gkIuT9omypnsUnzmtnkUkyn4aYGqg/viewform?usp=dialog) to get the password.

Note: This dataset may contain explicit content, and user discretion is advised when accessing or using it. 

- Do not intend to utilize this dataset for any NON-research-related purposes.
- Do not intend to distribute or publish any segments of the data.
- Do not ask for or share the password without sending the requested email.


## Search adversarial prompt:

``python main.py --target='sd' --method='rl' --reward_mode='clip' --threshold=0.26 --len_subword=10 --q_limit=60 --safety='ti_sd'``

- You can change the parameters follow the choices in ``main.py``. The adversarial prompts and statistic results (xx.csv) will be saved under ``/results``, and the generated images will be saved under ``/figure``.
- e.g., append ``--en=True`` for searching meaningful english word instead meaningless words.

### 11.2024 Update:

Add three Stable Diffusion models trained with safety alignment. Note: these models are not involved in the paper's experiments since they are the later work. Our paper only test the SD-1.4.
- no added ``--align`` or ``--align=no_align``: Our paper's default SD-1.4
- ``--align='sdxl'``: [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952). Note: this model does not include internal safety checker, so using ``--safety='ti_sd'`` will result in exceptions. However, you can try other add-on safety filters listed in ``main.py``.
- ``--align='mace'``: [[CVPR 2024] MACE: Mass Concept Erasure in Diffusion Models](https://github.com/Shilin-LU/MACE). Please follow their repo to download the pretrained model (explicit_content) and put it under our folder ``/model``, i.e., '/model/explicit_content/erase_explicit_content/..(other folders)...model_index.json'. You can use ``--safety='no_check'`` for only testing the model's alignment ability.
- ``--align='safegen'``: [[CCS 2024] SafeGen: Mitigating Sexually Explicit Content Generation in Text-to-Image Models](https://github.com/LetterLiGo/SafeGen_CCS2024). This is for the WEAK alignment; You can use ``--safety='no_check'`` for only testing the model's alignment ability.

## Evaluate the result:

``python evaluate.py --path='PATH OF xx.csv'``

## Citation:

Please cite our paper if you find this repo useful.

```
@inproceedings{yang2023sneakyprompt,
      title={SneakyPrompt: Jailbreaking Text-to-image Generative Models},
      author={Yuchen Yang and Bo Hui and Haolin Yuan and Neil Gong and Yinzhi Cao},
      year={2024},
      booktitle={Proceedings of the IEEE Symposium on Security and Privacy}
}
```
