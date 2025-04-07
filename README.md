<h1 align="center"> Screen Perturbation: Adversarial Attack and Defense on Under-Screen Camera</h1>

This repository is the official implementation of "[Screen Perturbation: Adversarial Attack and Defense on Under-Screen Camera](https://doi.org/10.1145/3570361.3613278)"
in the ACM Conference on Mobile Computing And Networking (MobiCom) 2023 
![Generic badge](https://img.shields.io/badge/code-official-green.svg)

![overview1](./Overview1.png)

## Introduction
Smartphones are moving towards the fullscreen design for better user experience. This trend forces front cameras to be placed under screen, leading to Under-Screen Cameras (USC). Accordingly, a small area of the screen is made translucent to allow light to reach the USC. In this paper, we utilize the translucent screen's features to inconspicuously modify its pixels, imperceptible to human eyes but inducing perturbations on USC images. These screen perturbations affect deep learning models in image classification and face recognition. They can be employed to protect user privacy, or disrupt the front camera's functionality in the malicious case. We design two methods, one-pixel perturbation and multiple-pixel perturbation, that can add screen perturbations to images captured by USC and successfully fool various deep learning models. Our evaluations, with three commercial full-screen smartphones on testbed datasets and synthesized datasets, show that screen perturbations significantly decrease the average image classification accuracy, dropping from 85% to only 14% for one-pixel perturbation and 5.5% for multiple-pixel perturbation. For face recognition, the average accuracy drops from 91% to merely 1.8% and 0.25%, respectively.

## Dependencies
Please install the required packages first by executing the command below:
```
pip install -r requirements.txt
```

## Embed screen perturbation to the existing dataset
We built the under-screen camera imaging model to evaluate the effect of proposed screen perturbation on the deep learning models' decisions. For image classification task, please download the miniImageNet dataset, and run the command to generate corresponding screen perturbations:
```
python add_perturbation_synthesized.py
```
If you want to use a potential image quality restoration algorithm to ensure the effect of screen perturbation, run the command
```
python add_perturbation_synthesized_deblurred.py
```
For face recogniztion task, please download the miniImageNet dataset, and run the command to generate corresponding screen perturbations:
```
python add_perturbation_synthesized_face.py
```
```
python add_perturbation_synthesized_face_deblurred.py
```

## Embed screen perturbation to the practical captured dataset
We setup a testbed to acquire practical USC images. The setup is shown below, which consists of a 4K LCD monitor displaying pristine images, and three COTS USC smartphones, i.e., Samsung Fold4, ZTE AXON30, and Xiaomi MIX4. The screenshots of the status bar when the three smartphones are generating one-pixel perturbation. As highlighted in the red dotted rectangle, the changes of screen-pixel units in the TSR are imperceptible. Meanwhile, the resulting screen perturbations differ as different screen layouts on these smartphones.

![overview2](./Overview2.png)

All high-resolution full-face images displayed on the 4K monitor are from the XGaze dataset. You can also run the following commands to embed screen perturbation on practical captured images:
```
python add_perturbation_smartphone_onepixel.py
```
```
python add_perturbation_smartphone_allpixels.py
```
For the practical captured images without screenn perturbations in the baseline, you can run the following commend:
```
add_perturbation_smartphone_displayoff.py
```

All data captured by our built testbed can be downloaded here (3.2G).


## Citation

If our work is useful for your research, please consider citing it:

```
@inproceedings{ye2023screen,
  title={Screen Perturbation: Adversarial Attack and Defense on Under-Screen Camera},
  author={Ye, Hanting and Lan, Guohao and Jia, Jinyuan and Wang, Qing},
  booktitle={Proceedings of the ACM Conference on Mobile Computing And Networking (MobiCom)},
  year={2023}
}
```