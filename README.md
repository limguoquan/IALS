# **I**nstance-**A**ware **L**atent-Space **S**earch

<img src='image\result.gif'/>
<!-- ![avatar](image\result.gif) -->

This is a PyTorch implementation of the following paper:

**Disentangled Face Attribute Editing via Instance-Aware Latent Space Search**, IJCAI 2021.

Yuxuan Han, Jiaolong Yang and Ying Fu

Paper: comming soon.

**Abstract**: *Recent works have shown that a rich set of semantic directions exist in the latent space of Generative Adversarial Networks (GANs), which enables various facial attribute editing applications.
However, existing methods may suffer poor attribute variation disentanglement, leading to unwanted change of other attributes when altering the desired one. 
The semantic directions used by existing methods are at attribute level, which are difficult to model complex attribute correlations, especially in the presence of attribute distribution bias in GAN’s training set. 
In this paper, we propose a novel framework (**IALS**) that performs **I**nstance-**A**ware **L**atent-Space **S**earch to find semantic directions for disentangled attribute editing. The instance information is injected by leveraging the supervision from a set of attribute classifiers evaluated on the input images. 
We further propose a Disentanglement-Transformation (DT) metric to quantify the attribute transformation and disentanglement efficacy and find the optimal control factor between attribute-level and instance-specific directions based on it. Experimental results on both GAN-generated and real-world images collectively show that our method outperforms state-of-the-art methods proposed recently by a wide margin.*

## Requirements
It's quite easy to create the environment for our model, you only need:
* Python 3.7 and the basic Anaconda3 environment. 
* PyTorch 1.x with GPU support (a single NVIDIA GTX 1060 is enough).
* The tqdm library to visualize the progress bar.

## Reproduce Results
Download the ```pretrain``` directory from [here](https://drive.google.com/file/d/1kpX9G9RNjJjdxbRwVnuk3SV8fvbCS-RV/view?usp=sharing) and put it on the root directory of this repository. If your environment meets our requirements, you will see an editing result in ```test_env.jpg``` using the following command. 

```
python edit_single_attr.py --seed 0 --step 0.5 --n_steps 4 --dataset ffhq --base interfacegan --attr male --save_path test_env.jpg
```

* Edit a random image generated by StyleGAN. You can specify the primal and condition attributes and the seed. Here we set gender as the primal attribute and expression as the condition attribute.
```
# reproduce our results:
python condition_manipulation.py --seed 0 --step 0.1 --n_steps 30 --dataset ffhq --base interfacegan --attr1 male --attr2 smiling --lambda1 0.75 --lambda2 0 --real_image 0 --save_path rand-ours.jpg

# reproduce interfacegan results:
python condition_manipulation.py --seed 0 --step 0.1 --n_steps 30 --dataset ffhq --base interfacegan --attr1 male --attr2 smiling --lambda1 1 --lambda2 1 --real_image 0 --save_path rand-inter.jpg
```

* Edit a real face image via our instance-aware direction. In the ```pretrain\real_latent_code``` folder we put lots of pretrained latent code provided by [seeprettyface](http://www.seeprettyface.com/index_page6.html). If you want to edit customized face images, please refer to the next section.
**Note**: *If lambda1=lambda2=1, our method degrades to the attribute-level semantic direction based methods like InterfaceGAN and GANSpace*.

```
# reproduce our results:
python condition_manipulation.py --seed 0 --step -0.1 --n_steps 30 --dataset ffhq --base interfacegan --attr1 young --attr2 eyeglasses --lambda1 0.75 --lambda2 0 --real_image 1 --latent_code_path pretrain\real_latent_code\real1.npy --save_path real-ours.jpg

# reproduce interfacegan results: 
python condition_manipulation.py --seed 0 --step -0.1 --n_steps 30 --dataset ffhq --base interfacegan --attr1 young --attr2 eyeglasses --lambda1 1 --lambda2 1 --real_image 1 --latent_code_path pretrain\real_latent_code\real1.npy --save_path real-inter.jpg
```

* Compute the attribute-level direction by average the instance-specific direction.
```
python train_attr_level_direction.py --n_images 500 --attr pose
```

## Editing Your Own Image
Typically you need to follow the steps below:
1. Obtain the latent code of the real image via GAN Inversion. Here we provide a simple baseline GAN-Inversion method in ```gan_inversion.py```. 
```
python gan_inversion.py --n_iters 500 --img_path image\real_face_sample.jpg
```

2. Editing the real face image's latent code with our method. 
```
python condition_manipulation.py --seed 0 --step -0.1 --n_steps 10 --dataset ffhq --base interfacegan --attr1 male --attr2 smiling --lambda1 0.75 --lambda2 0 --real_image 1 --latent_code_path rec.npy --save_path real-ours.jpg
```

You will see the result like that:

<img src='image\real_face_edit.jpg'/>
<!-- ![avatar](image\real_face_edit.jpg) -->

To improve the editing quality, we highly recommand you to use the state-of-the-art GAN inversion method like [Id-Invert](https://github.com/genforce/idinvert) or [pixel2image2pixel](https://github.com/eladrich/pixel2style2pixel). 
**Note**: *You need to make sure that these GAN inversion methods use the same pretrained StyleGAN weights as us.*

## Contact
If you have any questions, please contact Yuxuan Han (hanyuxuan@bit.edu.cn).

## Citation
Please cite the following paper if this model helps your research:

    @inproceedings{han2021IALS,
	    title={Disentangled Face Attribute Editing via Instance-Aware Latent Space Search},
	    author={Yuxuan Han, Jiaolong Yang and Ying Fu},
        booktitle={International Joint Conference on Artificial Intelligence},
        year={2021}
    }

## Acknowledgments
This code borrows the StyleGAN generator implementation from https://github.com/lernapparat/lernapparat and uses the pretrained real image's latent code provided by http://www.seeprettyface.com/index_page6.html. We thank for their great effort! 
