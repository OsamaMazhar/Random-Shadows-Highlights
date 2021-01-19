# Random Shadows and Highlights


<p align="center">
  <img src="./Samples/RSH_0.gif" width="300" />
  <img src="./Samples/RSH_1.gif" width="300" />
</p>
<p align="center">
  <img src="./Samples/RSH_2.gif" width="300" />
  <img src="./Samples/RSH_3.gif" width="300" />
</p>

This repo has the source code for the paper: [Random Shadows and Highlights: A new data augmentation method for extreme lighting conditions](https://arxiv.org/abs/2101.05361).

### Example:
```
from RandomShadowsHighlights import RandomShadows

 transform = transforms.Compose([
   transforms.RandomHorizontalFlip(),
   RandomShadows(p=0.8, high_ratio=(1,2), low_ratio=(0,1), left_low_ratio=(0.4,0.8),
                 left_high_ratio=(0,0.3), right_low_ratio=(0.4,0.8), right_high_ratio=(0,0.3)),
   transforms.ToTensor(),
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
 ])
 ```

If you find this code useful for your research, please consider citing:

    @article{mazhar2021rsh,
    title={Random Shadows and Highlights: A new data augmentation method for extreme lighting conditions},
    author={Osama Mazhar and Jens Kober},
    journal={arXiv preprint arXiv:2101.05361},
    year={2021}
    }

### Requirements:
```torch, torchvision, numpy, cv2, PIL, argparse```

In case you want to use <em>Disk-Augmenter</em> for comparison, then install ```scikit-learn``` as well.

### Steps:
To test on **TinyImageNet**, the dataset needs to be converted into PyTorch dataset format. This can be done by following instructions on this [repo](https://github.com/tjmoon0104/pytorch-tiny-imagenet).

Also, for **EfficientNet**, install EfficientNet-PyTorch from [here](https://github.com/lukemelas/EfficientNet-PyTorch).

To start training, use the following command:

```python main.py --model_dir outputs --filename output.txt --num_epochs 20 --model_name EfficientNet --dataset TinyImageNet```

For **CIFAR10** or **CIFAR100**, use argument ```--dataset CIFAR10``` or ```--dataset CIFAR100```.

To train on "AlexNet", use ```--model_name AlexNet```.

##

If you have any questions about this code, please do not hesitate to contact me [here](mailto:osamazhar@yahoo.com).
