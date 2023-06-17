# <div align="center"> Identity-driven Three-Player Generative Adversarial Network for Synthetic-based Face Recognition </div>

<div align="center">
  Authors: 
  <br>
  Jan Niklas Kolf, Tim Rieber, Jurek Elliesen, Fadi Boutros, Arjan Kuijper, Naser Damer
  <br>
  <br>
  <br>
  <a 
    href="https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Kolf_Identity-Driven_Three-Player_Generative_Adversarial_Network_for_Synthetic-Based_Face_Recognition_CVPRW_2023_paper.pdf">
    <img src="https://github.com/jankolf/assets/blob/main/IDnet/paper-thecvf.com.svg?raw=true" alt="Paper available at TheCVF">
  </a>
  <a 
    href="https://share.jankolf.de/s/L2G68G9WGXr6Tkt">
    <img src="https://github.com/jankolf/assets/blob/main/IDnet/data-download.svg?raw=true" alt="Data available to download"> 
  </a>
</div>  



## <div align="center"> Overview 🔎 </div>
<div align="center">
  <p>
    <img width="90%" src="https://github.com/fdbtrs/Synthetic-Face-Recognition/raw/master/images/overview_IDnet.png?raw=true">
  </p>
</div>

## <div align="center"> Abstract 🤏 </div>
Many of the commonly used datasets for face recognition development are collected from the internet without proper user consent. Due to the increasing focus on privacy in the social and legal frameworks, the use and distribution of these datasets are being restricted and strongly questioned. These databases, which have a realistically high variability of data per identity, have enabled the success of face recognition models. To build on this success and to align with privacy concerns, synthetic databases, consisting purely of synthetic persons, are increasingly being created and used in the development of face recognition solutions. In this work, we present a three-player generative adversarial network (GAN) framework, namely IDnet, that enables the integration of identity information into the generation process. The third player in our IDnet aims at forcing the generator to learn to generate identity-separable face images. We empirically proved that our IDnet synthetic images are of higher identity discrimination in comparison to the conventional two-player GAN, while maintaining a realistic intra-identity variation. We further studied the identity link between the authentic identities used to train the generator and the generated synthetic identities, showing very low similarities between these identities. We demonstrated the applicability of our IDnet data in training face recognition models by evaluating these models on a wide set of face recognition benchmarks. In comparison to the state-of-the-art works in synthetic-based face recognition, our solution achieved comparable results to a recent rendering-based approach and outperformed all existing GAN-based approaches. The training code and the synthetic face image dataset are publicly available.

## <div align="center"> Usage 🖥 </div>

### Train IDnet
To train the three-player GAN IDnet, 
1. download the offical StyleGAN2-ADA Pytorch implementation from the [Github repository](https://github.com/NVlabs/stylegan2-ada-pytorch)
2. prepare your training dataset
   ```
   python dataset_tool.py --source=/data/folder --dest=/data/GAN_datasets/dataset.zip --width=128 --height=128
   ```
3. clone this repository, copy the content of IDnet folder into StyleGAN2-ADA
4. Download the pre-trained backbone model for **ID-3** [from this link](https://share.jankolf.de/s/jQAF93YfCmbpePL)
5. Enter the path to the downloaded pretrained backbone of **ID-3** in training_loop.py:182
6. Enter the number of labels in training_loop.py:170 (default: 10572)
7. To run training, execute
   ```
   python train.py --outdir=./output/modelname --gpus=4 --data=/data/GAN_datasets/dataset.zip --cond=1
   ```

### Synthesize Images
To create synthetic images, either train a generator or [download our generator from this link](https://share.jankolf.de/s/g764N8JnZ5LBoT7) and
1. Prepare codebase (see **> Train IDnet** section)
2. Save generator in the source folder
3. Execute
   ```
   python generate_images.py
   ```
4. Create an image list file (image_list.txt) with content \<path/to/image class-label> per line:
   ```
   000000/000000_normal_001.jpg	0
   ```

### Train Face Recognition Model
To train a face recognition model,
1. Create a new synthetic dataset or [download our dataset from this link](https://share.jankolf.de/s/wLDgpLmRWWtBFnd)
1. Clone/download the FaceRecognition folder from this repository
2. Run train.py with the path to the image_list.txt:
   ```
   python train.py --samples image_list.txt
   ```

## <div align="center"> Citation ✒ </div>
If you found this work helpful for your research, please cite the article with the following bibtex entry:
```
@InProceedings{Kolf_2023_CVPR,
    author    = {Kolf, Jan Niklas and Rieber, Tim and Elliesen, Jurek and Boutros, Fadi and Kuijper, Arjan and Damer, Naser},
    title     = {Identity-Driven Three-Player Generative Adversarial Network for Synthetic-Based Face Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {806-816}
}
```

