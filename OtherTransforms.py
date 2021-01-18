import random
import torchvision.transforms.functional as TF

class RandomGamma(object):
    def __init__(self, gamma_p = 0.5, gamma_ratio=(0,1.5)):
        self.gamma_p = gamma_p
        self.gamma_ratio = gamma_ratio

    def __call__(self,img):
        if random.uniform(0, 1) < self.gamma_p:
            gamma = random.uniform(self.gamma_ratio[0], self.gamma_ratio[1])
            img = TF.adjust_gamma(img, gamma, gain=1)
            return img
        else:
            return img

class RandomColorJitter(object):
    def __init__(self, p = 0.5, brightness_ratio=(0,2), contrast_ratio=(0,2), \
                saturation_ratio=(0,2), hue_ratio=(-0.5,0.5)):
        self.p = p
        self.brightness_ratio = brightness_ratio
        self.contrast_ratio = contrast_ratio
        self.saturation_ratio = saturation_ratio
        self.hue_ratio = hue_ratio

    @staticmethod
    def process(img, brightness_ratio, contrast_ratio, saturation_ratio, hue_ratio):
        brightness = random.uniform(brightness_ratio[0], brightness_ratio[1])
        contrast = random.uniform(contrast_ratio[0], contrast_ratio[1])
        saturation = random.uniform(saturation_ratio[0], saturation_ratio[1])
        hue = random.uniform(hue_ratio[0], hue_ratio[1])

        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_saturation(img, saturation)
        img = TF.adjust_hue(img, hue)

        return img

    def __call__(self,img):
        if random.uniform(0, 1) < self.p:
            img = self.process(img, self.brightness_ratio, self.contrast_ratio, \
                                self.saturation_ratio, self.hue_ratio)
            return img
        else:
            return img
