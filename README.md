# 🦙🌄 Multimodal (Visual) Instruction Tuning for LLaMA 3

This repository contains code for multimodal (visual) instruction tuning of the LLaMA 3 language model. 

The idea is to fine-tune the LLaMA 3 model on a multimodal dataset that contains both textual instructions and visual demonstrations. We trained this model with the [llava_instruct_80k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_80k.json) dataset. See [LLaVA](https://llava-vl.github.io/) for related methods.

The code is intended to be an easy-to-use starting point for multimodal instruction tuning. Uses [Hugging Face Transformers](https://huggingface.co/) library for the LLaMA 3 model and for training.

<p align="center">
  <img src="https://github.com/AdrianBZG/llama-multimodal-vqa/assets/8275330/256fec3b-fb2a-440c-871f-a2f92953a03c">
</p>


## Usage

To fine-tune the LLaMA 3 model on your own multimodal dataset, you need to prepare the dataset and then run the training script.

### Prepare Dataset

The dataset should be in the format of a JSON file with entries for conversations (textual instructions) and image URIs (visual condition). See [this](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) for an example.

For example, the dataset could look like this:

```
[
{"conversations": [ { "from": "human", "value": "<image>\nWhat skill set might someone need to perform such a frisbee trick?" }, { "from": "gpt", "value": "To perform the frisbee trick shown in the image, where the man is passing a frisbee between or underneath his legs, a person would need a combination of skills. These skills include good hand-eye coordination, agility, balance, flexibility, and dexterity. Additionally, the ability to throw and catch the frisbee accurately while maintaining control of bodily movements would also be essential. To perfect the trick, practicing these skills and building up muscle memory through repetition would be beneficial." } ], "image": "000000215677.jpg"},
{"conversations": [ { "from": "human", "value": "<image>\nWhat precautions is the woman taking while walking in this weather?" }, { "from": "gpt", "value": "The woman is taking precautions while walking in the rainy weather by using an open umbrella to shield herself from the rain. The umbrella helps to keep her dry and protected from getting soaked by the rain, ensuring her comfort and safety while walking down the wet street. Holding the umbrella also indirectly indicates that the woman might be more attentive to her surroundings and cautious of potential hazards caused by the wet conditions, such as slippery surfaces, puddles, or splashing from passing vehicles." } ], "image": "000000296754.jpg"}
]
```

### Run Training

To fine-tune the LLaMA 3 model on the multimodal dataset, run the training script with the following command:

```bash
python train.py --dataset_path path/to/dataset.json --output_dir path/to/output_dir --text_model_id="meta-llama/Meta-Llama-3-8B-Instruct" --vision_model_id="openai/clip-vit-large-patch14" --batch_size 32
```

This will fine-tune the LLaMA 3 model on the dataset and save the fine-tuned model to the output directory.

## Acknowledgements

This code is based on the [Hugging Face Transformers](https://huggingface.co/transformers/) library and the [LLaVA](https://llava-vl.github.io/) project.

## Useful Links

- [LLaVA](https://llava-vl.github.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [LLaMA 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [LLaMA 3 70B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
