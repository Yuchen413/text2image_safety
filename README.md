# SneakyPrompt: Jailbreaking Text-to-image Generative Models

This is the official implementation of the paper "[SneakyPrompt: Jailbreaking Text-to-image Generative Models](https://arxiv.org/abs/2305.12082)".

## Search adversarial prompt:

``python main.py --target='sd' --method='rl' --reward_mode='clip' --threshold=0.26 --len_subword=10 --q_limit=60 --safety='ti_sd'``

You can change the parameters follow the choices in 'search.py'. The adversarial prompts and statistic results (xx.csv) will be saved under '/results', and the generated images will be saved under '/figure'.

## Evaluate the result:

``python evaluate.py --path='PATH OF xx.csv'``

## Citation:

Please cite our paper, if you find this repo useful.

```
@inproceedings{yang2023sneakyprompt,
      title={SneakyPrompt: Jailbreaking Text-to-image Generative Models},
      author={Yuchen Yang and Bo Hui and Haolin Yuan and Neil Gong and Yinzhi Cao},
      year={2023},
      booktitle={Proceedings of the IEEE Symposium on Security and Privacy (Oakland)}
}
```
