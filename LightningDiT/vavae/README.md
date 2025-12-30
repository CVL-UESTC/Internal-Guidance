## Training Scripts of VA-VAE

### Installation

1. Install the lightningdit environment first.

2. Install additional packages:
    ```
    pip install -r vavae_requirements.txt
    ```

3. [Taming-Transformers](https://github.com/CompVis/taming-transformers?tab=readme-ov-file) is also needed for training. 
    
    Get it by running:
    ```
    git clone https://github.com/CompVis/taming-transformers.git
    cd taming-transformers
    pip install -e .
    ```

    Then modify ``./taming-transformers/taming/data/utils.py`` to meet torch 2.x:
    ```
    export FILE_PATH=./taming-transformers/taming/data/utils.py
    sed -i 's/from torch._six import string_classes/from six import string_types as string_classes/' "$FILE_PATH"
    ```


### Train

1. Modify training config as you need.

2. Run training by:

    ```
    bash run_train.sh vavae/configs/f16d32_vfdinov2.yaml
    ```
    Your training logs and checkpoints will be saved in the `logs` folder. We train VA-VAE with 4x8 H800 GPUs.

### Evaluate

1. We provide a training log [here](https://huggingface.co/hustvl/va-vae-imagenet256-experimental-variants/tree/main/tensorboard_logs) for reference. All of our experimental variants are provided [here](https://huggingface.co/hustvl/va-vae-imagenet256-experimental-variants/tree/main).

2. Put your checkpoint path into ``lightningdit/tokenizer/configs/vavae_f16d32.yaml`` and use ``lightningdit/evaluate_tokenizer.py`` to evaluate the model.

### Acknowledgement

VA-VAE's training is mainly built upon [LDM](https://github.com/CompVis/latent-diffusion/tree/main). Thanks for the great work!
