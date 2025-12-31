
## Guiding a Diffusion Transformer with the Internal Dynamics of Itself (IG)<br><sub>Official PyTorch Implementation</sub>

### [Paper]() | [Project Page](https://zhouxingyu13.github.io/Internal-Guidance/) | [Models](https://huggingface.co/CVLUESTC/Internal-Guidance) 

> [**Guiding a Diffusion Transformer with the Internal Dynamics of Itself**]()<br>
> [Xingyu Zhou](https://zhouxingyu13.github.io/)¹, [Qifan Li](https://scholar.google.com/citations?user=1ssHRA8AAAAJ&hl=zh-CN)¹, [Xiaobin Hu](https://huuxiaobin.github.io/)²,  [Hai Chen](https://openreview.net/profile?id=%7EHai_Chen3)<sup>3,4</sup>,  [Shuhang Gu](https://shuhanggu.github.io/)¹*
> <br><sup>1</sup>University of Electronic Science and Technology of China <sup>2</sup>National University of Singapore<br>
> <sup>3</sup>Sun Yat-sen University <sup>4</sup>North China Institute of Computer Systems Engineering<br>
> <sup>*</sup>Corresponding Author
> 
![LightningDiT+IG samples](visual.png)


### 💥 News  
- **[2025.12.31]** We have released the paper and code of IG.


### 🌟 Highlight
 -  **🔥New SOTA on 256 &times; 256 ImageNet generation:** LightningDiT-XL/1 + IG sets a new state of the art with <strong>FID = 1.07</strong> (random sampling FID = 1.19) on ImageNet, while achieving FID = 1.24 (random sampling FID = 1.34) without classifier-free guidance.

-  **Simple enough, powerful enough:**  We present <strong>Internal Guidance (IG)</strong>, a simple yet powerful guidance mechanism for Diffusion Transformers. Just requiring an additional intermediate supervision is all that is needed.

- **intermediate supervision:** A simple intermediate supervision can achieve a similar effect to the additional designed self-supervised learning regularization.

- **Improved Performance:** IG accelerates training and improves generation performance for DiTs, SiTs and LightningDiT.

## 📝 Results
- State-of-the-art Performance on ImageNet 256x256 with FID=1.19 (random sampling).
![Random sampling](randomsota.png)
- State-of-the-art Performance on ImageNet 256x256 with FID=1.07 (uniform balanced sampling).
![Uniform balanced sampling](unisota.png)
### 🏡 Environment Setup

```bash
conda create -n IG python=3.12 -y
conda activate IG
pip install -r requirements.txt
```

### 📜 Dataset Preparation

Currently, we provide experiments for [ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). You can place the data that you want and can specify it via `--data-dir` arguments in training scripts. \
Note that we preprocess the data for faster training. Please refer to [preprocessing guide](https://github.com/CVL-UESTC/Internal-Guidance/tree/main/SiT/preprocessing) for SiTs and [README.md](https://github.com/CVL-UESTC/Internal-Guidance/tree/main/LightningDiT/README.md) for LightningDiTs for detailed guidance.

### 🔥 Training
Here we provide the training code for SiTs and LightningDiTs.

##### 5.1.Training with SiT + IG
```bash
cd SiT
accelerate launch --config_file configs/default.yaml train.py \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --resolution=256 \
  --batch-size=32 \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="sitxl-ab820-t0.2-res256" \
  --data-dir=[YOUR_DATA_PATH]
```

Then this script will automatically create the folder in `exps` to save logs,samples, and checkpoints. You can adjust the following options:

- `--models`: Choosing from [SiT-B/2, SiT-L/2, SiT-XL/2]
- `--encoder-depth`: Intermediate output block layer for the auxiliary supervision
- `--output-dir`: Any directory that you want to save checkpoints, samples, and logs
- `--exp-name`: Any string name (the folder will be created under `output-dir`)
- `--batch-size`: The local batch size (by default we use 1 node of 8 GPUs), you need to adjust this value according to your GPU number to make total batch size of 256


##### 5.2.Training with LightningDiT + SRA
```bash
cd LightningDiT
bash run_train.sh configs/lightningdit_xl_vavae_f16d32.yaml
```

Then this script will automatically create the folder in `output` to save logs and checkpoints. You can adjust the following options by the original [LightningDiT](https://github.com/hustvl/LightningDiT).



### 🌠 Evaluation
Here we provide the generating code (**random sampling**) for SiTs and LightningDiTs to get the samples for evaluation. (and the .npz file can be used for [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations) suite) through the following script:

You can download our pretrained model here:

| Model                   | Image Resolution | Epochs  | FID-50K | Inception Score |
|-------------------------|------------------| --------|---------|-----------------|
| [SiT-XL/2 + IG](https://huggingface.co/CVLUESTC/Internal-Guidance/tree/main/SiT) | 256x256          |  800    | 1.46    |   265.7       |
| [LightningDiT-XL/1 + IG](https://huggingface.co/CVLUESTC/Internal-Guidance/tree/main/Lightningdit) | 256x256          |  680    | 1.19    |   269.0        |
##### Sampling with SiT + IG
```bash
cd SiT
bash gen.sh
```
Note that there are several options in `gen.sh` file that you need to complete:
- `SAMPLE_DIR`: Base directory to save the generated images and .npz file
- `CKPT`: Checkpoint path (This can also be your downloaded local file of the ckpt file we provide above)


##### Sampling with LightningDiT + IG
```bash
cd LightningDiT
bash run_inference.sh configs/lightningdit_xl_vavae_f16d32.yaml
```


### 📣 Note

It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release as well as the difference of the hardware facility. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. 

### 🤝🏻 Acknowledgement

This code is mainly built upon [SRA](https://github.com/vvvvvjdy/SRA), [LightningDiT](https://github.com/hustvl/LightningDiT), [RAE](https://github.com/bytetriper/RAE) repositories. 
Thanks for their solid work!


