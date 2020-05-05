from albumentations import *

p=0.6
M=6
shape=(224,224)

aug_pool = [HueSaturationValue(hue_shift_limit=(-10.0-M*3),sat_shift_limit=0,
    val_shift_limit=0, p=p),
    ElasticTransform(p=p, approximate=True, alpha=0, sigma=1+0.5*M,
        alpha_affine=10+4*M),
    RandomSizedCrop((shape[0]-M*11,shape[1]-M*11), shape[0], shape[1], p=p),
    RandomContrast(limit=0+M*0.1, p=p),
    RandomBrightness(limit=0+M*0.15, p=p),
    Blur(blur_limit=0+M*0.5, p=p),
    RandomGamma(gamma_limit=(50+M*10, 80+M*10), p=p),
    GridDistortion(num_steps=5, distort_limit=0+M*0.04, interpolation=1,
        border_mode=4, p=p)]

