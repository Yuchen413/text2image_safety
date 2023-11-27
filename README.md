# SneakyPrompt: Jailbreaking Text-to-image Generative Models

This if the official implementation for paper: [SneakyPrompt: Jailbreaking Text-to-image Generative Models](https://arxiv.org/abs/2305.12082)

Our work has been reported by [MIT Technology Review](https://www.technologyreview.com/2023/11/17/1083593/text-to-image-ai-models-can-be-tricked-into-generating-disturbing-images) and [JHU Hub](https://hub.jhu.edu/2023/11/01/nsfw-ai/). Please check them out if interested.


## Environment setup

The experiment is run on Ubuntu 18.04, with one Nvidia 3090 GPU (24G). Please install the dependencies via:

``conda env create -f environment.yml``


## Dataset

The [nsfw_200.txt](https://livejohnshopkins-my.sharepoint.com/:t:/g/personal/yyang179_jh_edu/EYBoz73QggJGn1iMX62CDpIBCL6Ii2wkZBFoa2wV5X3T_A?e=9G8nar) can be access per request, please send the author an email for password.

Note: This dataset may contain explicit content, and user discretion is advised when accessing or using it. 

- Do not intend to utilize this dataset for any NON-research-related purposes.
- Do not intend to distribute or publish any segments of the data.


## Search adversarial prompt:

``python main.py --target='sd' --method='rl' --reward_mode='clip' --threshold=0.26 --len_subword=10 --q_limit=60 --safety='ti_sd'``

You can change the parameters follow the choices in 'search.py'. The adversarial prompts and statistic results (xx.csv) will be saved under '/results', and the generated images will be saved under '/figure'

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