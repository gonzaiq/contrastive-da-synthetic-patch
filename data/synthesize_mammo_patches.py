import numpy as np
import os
import pandas as pd
from PIL import Image
import json
import copy
from scipy import ndimage
import argparse

parser = argparse.ArgumentParser(description="Generate synthetic patches with random parameters.")

# Define arguments
parser.add_argument("--n_imgs", type=int, default=1000, help="Number of images to generate")
parser.add_argument("--beta_range", type=float, nargs=2, default=[1.2, 1.6], help="Range for beta values")
parser.add_argument("--calc_area_len_range", type=int, nargs=2, default=[15, 60], help="Range for calcification area lengths")
parser.add_argument("--n_calcs_range", type=int, nargs=2, default=[5, 12], help="Range for number of calcifications")
parser.add_argument("--calc_intensity_range", type=float, nargs=2, default=[0.3, 0.3], help="Range for calcification intensity")
parser.add_argument("--calc_radius_range", type=int, nargs=2, default=[1, 1], help="Range for calcification radius")
parser.add_argument("--mass_intensity_range", type=float, nargs=2, default=[0.3, 0.3], help="Range for mass intensity")
parser.add_argument("--mass_radius_range", type=int, nargs=2, default=[5, 45], help="Range for mass radius")
parser.add_argument("--add_noise", action="store_true", help="Whether to add noise")
parser.add_argument("--sigma_noise", type=float, default=0.1, help="Standard deviation of noise")
parser.add_argument("--img_w", type=int, default=256, help="Width of the generated images")
parser.add_argument("--img_h", type=int, default=256, help="Height of the generated images")
parser.add_argument("--p_calc", type=float, default=0.3, help="Probability of calcifications")
parser.add_argument("--p_mass", type=float, default=0.3, help="Probability of mass")
parser.add_argument("--p_breast_border", type=float, default=0.5, help="Probability of breast border")
parser.add_argument("--save_dir", type=str, default="data/synthetic_patches_new", help="Directory to save the generated data")
parser.add_argument("--out_annots_fname", type=str, default="annots.csv", help="Output annotations filename")

# Randomly flip an array along vertical and horizontal axes
def randomly_flip_array(_array):

    # Flip along vertical axis with 50% probability
    if np.random.rand() < 0.5:
        _array = np.flip(_array, axis=0)

    # Flip along horizontal axis with 50% probability
    if np.random.rand() < 0.5:
        _array = np.flip(_array, axis=1)

    return _array

# Randomly rotate an array by 0, 90, 180, or 270 degrees
def randomly_rotate_array(_array):

    prob = np.random.rand()
    if prob < 0.25:
        # No rotation
        pass
    elif prob < 0.5:
        # Rotate by 90 degrees
        _array = np.rot90(_array, axes=(0, 1))
    elif prob < 0.75:
        # Rotate by 180 degrees
        _array = np.rot90(_array, axes=(0, 1), k=2)
    else:
        # Rotate by 270 degrees
        _array = np.rot90(_array, axes=(0, 1), k=3)

    return _array

# Generate a random breast-shaped mask for the image
def generate_random_breast_mask(img):

    img_h = img.shape[0]
    img_w = img.shape[1]

    # Ensure square input image and dimensions of 256 or 512
    assert img_h == img_w
    assert img_h in [256, 512]

    # Define mask parameters based on image size
    if img_h == 512:
        x0 = (100 - 30) * np.random.rand() + 30
        y0 = (100 - 30) * np.random.rand() + 30
        a = 300  # Radius along x-axis
        b = 400  # Radius along y-axis
    elif img_h == 256:
        x0 = (100 - 30) * np.random.rand() + 30
        y0 = (100 - 30) * np.random.rand() + 30
        a = 200  # Radius along x-axis
        b = 300  # Radius along y-axis

    alpha = (0.6 - 0.4) * np.random.rand() + 0.4  # Rotation angle

    # Compute rotation parameters
    _cos = np.cos(alpha)
    _sin = np.sin(alpha)

    # Create a grid for ellipse computation
    X, Y = np.mgrid[-int(img_h/2):int(img_h/2), -int(img_w/2):int(img_w/2)]

    # Define ellipse mask
    mask = (((_cos*X+_sin*Y)-(_cos*x0+_sin*y0))/a)**2 + (((_sin*X-_cos*Y)-(_sin*x0-_cos*y0))/b)**2 <= 1

    # Randomly rotate the mask
    mask = randomly_rotate_array(mask)

    # Randomly flip the mask
    mask = randomly_flip_array(mask)

    return mask

# Generate breast texture based on beta parameter
def generate_breast_texture(beta, img_w, imgh_h):

    scaling_factor = np.sqrt(img_w * imgh_h)

    # Generate random noise for texture
    We = np.random.randn(imgh_h, img_w)

    # Apply Fourier transform to noise
    W = np.fft.fft2(We)

    # Create frequency grid
    X, Y = np.mgrid[-int(imgh_h/2):int(imgh_h/2), -int(img_w/2):int(img_w/2)]

    # Compute filter based on beta
    FU = np.minimum(scaling_factor**beta, 1/((np.sqrt((X/imgh_h)**2 + (Y/img_w)**2))**beta + 10**(-6)))
    FU = np.fft.fftshift(FU)

    # Normalize the filter
    FU = (FU - FU.min()) / (FU.max()-FU.min())

    # Apply inverse Fourier transform to generate texture
    M = np.real(np.fft.ifft2(np.multiply(FU, W)))

    # Normalize texture
    M = (M-M.min()) / (M.max()-M.min())

    return M

def validate_lesion(lesion_type, lesion, mask):

    assert lesion_type in ['mass', 'calc']
    lesion_mask = copy.deepcopy(lesion)

    if lesion_type == "mass":
        # threshold mass
        lesion_mask[lesion_mask < np.exp(-3)] = 0
        lesion_mask[lesion_mask >= np.exp(-3)] = 1
    else:
        lesion_mask[lesion_mask > 0] = 1
        lesion_mask[lesion_mask <= 0] = 0

    if np.sum(lesion[mask == 0]) > 0.1:
        # lesions should not overlap with background
        return False 

    if np.sum(lesion[mask == 1]) == 0:
        # lesions should be contained in the breast
        return False 
    
    return True

def recursively_generate_lesion(
    lesion_type, 
    img_w, 
    img_h,
    calc_area_len=10, 
    n_calcs=10, 
    lesion_intensity=0.5, 
    lesion_radius=50, 
    breast_mask=None,
    max_its=20,
    oval_mass=False
    ):

    assert lesion_type in ['mass', 'calc']

    for _ in range(max_its):
        
        if lesion_type == "mass":
            _lesion = generate_mass(
                            img_w, 
                            img_h, 
                            mass_intensity=lesion_intensity, 
                            mass_radius=lesion_radius,
                            breast_mask=breast_mask,
                            oval_mass=oval_mass
                            )
            
        elif lesion_type == "calc":
            _lesion = generate_calc(
                            img_w, 
                            img_h, 
                            calc_area_len=calc_area_len,
                            n_calcs=n_calcs,
                            calc_intensity=lesion_intensity, 
                            calc_radius=lesion_radius,
                            breast_mask=breast_mask
                            )
    
        if validate_lesion(lesion_type, _lesion, breast_mask):
            return _lesion
    
    return None
    
def generate_mass(img_w, img_h, mass_intensity=0.5, mass_radius=50, breast_mask=None, oval_mass=False):

    X, Y = np.mgrid[-int(img_h/2):int(img_h/2), -int(img_w/2):int(img_w/2)]

    # get center of the mass
    if breast_mask is not None:
        # choose a random point in the breast, excluding point too near the borders
        breast_mask_modif = copy.deepcopy(breast_mask)
        breast_mask_modif[0:int(mass_radius/4), :] = 0
        breast_mask_modif[-int(mass_radius/4):, :] = 0
        breast_mask_modif[:, 0:int(mass_radius/4)] = 0
        breast_mask_modif[:, -int(mass_radius/4):] = 0

        # erode breast mask to avoid having masses centered too close to the breast borders
        breast_mask_modif = ndimage.binary_erosion(breast_mask_modif, iterations=int(mass_radius/2), border_value=0)

        rows, cols = np.where(breast_mask_modif==1)
        eligible_centers = [[x-int(img_h/2), y-int(img_w/2)] for x,y in zip(rows, cols)] # centers must be in centered coordinates

        pixel_idx = np.random.randint(0, len(eligible_centers)-1)
        center = eligible_centers[pixel_idx]
        Px = center[0]
        Py = center[1]
    else:
        # random position
        Px = np.random.choice(np.arange(-int(img_h/2), int(img_h/2)))
        Py = np.random.choice(np.arange(-int(img_w/2), int(img_w/2)))
    
    if oval_mass:
        # sample excentricity
        e = 0.9 * np.random.rand() # max excentricity will be 0.9

        if np.random.randn() < 0.5:
            x_radius = mass_radius # semi-major axis
            y_radius = mass_radius * np.sqrt(1-e**2) # semi-minor axis
        else:
            x_radius = mass_radius * np.sqrt(1-e**2) # semi-minor axis
            y_radius = mass_radius # semi-major axis
    else:
        x_radius = mass_radius
        y_radius = mass_radius

    mass = mass_intensity*(np.exp(-(((X-Px)/x_radius)**2 + ((Y-Py)/y_radius)**2)))

    return mass

def generate_calc(img_w, imgh_h, calc_area_len=50, n_calcs=10, calc_intensity=1.0, calc_radius=2, breast_mask=None):

    X, Y = np.mgrid[-int(imgh_h/2):int(imgh_h/2), -int(img_w/2):int(img_w/2)]

    calcs = np.zeros_like(X)

    # get center of the calcs
    if breast_mask is not None:
        # choose a random point in the breast
        rows, cols = np.where(breast_mask==1)
        eligible_centers = [[x,y] for x,y in zip(cols,rows)]

        pixel_idx = np.random.randint(0, len(eligible_centers)-1)
        center = eligible_centers[pixel_idx]
        Px = center[0]
        Py = center[1]
    else:
        # random position
        Px = np.random.choice(np.arange(-int(imgh_h/2), int(imgh_h/2)))
        Py = np.random.choice(np.arange(-int(img_w/2), int(img_w/2)))

    xidxs = np.arange(int(Px-calc_area_len/2), int(Px+calc_area_len/2))
    yidxs = np.arange(int(Py-calc_area_len/2), int(Py+calc_area_len/2))

    for _ in range(n_calcs):

        _px = np.random.choice(xidxs)
        _py = np.random.choice(yidxs)

        #calcs = calcs - calc_intensity * np.heaviside(((X-_px)**2 + (Y-_py)**2)-calc_radius, 1)
        calcs = calcs + calc_intensity * (((X-_px)**2 + (Y-_py)**2) < calc_radius**2)

        # remove used elements
        index = np.argwhere(xidxs==_px)
        xidxs = np.delete(xidxs, index)

        index = np.argwhere(yidxs==_py)
        xidxs = np.delete(yidxs, index)

    return calcs

def synthesize_patch(
        img_w,
        img_h,
        beta,
        add_noise,
        sigma_noise,
        add_mass,
        mass_intensity,
        mass_radius,
        add_calcs,
        calc_area_len,
        n_calcs,
        calc_intensity,
        calc_radius,
        save_dir,
        fname,
        add_borders
    ):

    img = generate_breast_texture(beta, img_w, img_h)

    if add_borders:
        mask = generate_random_breast_mask(img)
    else:
        mask = np.ones_like(img)

    per_bckg = 1 - np.sum(mask)/(img_h*img_w)

    img[mask == 0] = 0

    if add_noise:
        img = img + sigma_noise * np.random.randn(img_w, img_h)

    if add_mass:
        mass = recursively_generate_lesion(
            lesion_type="mass",
            img_w=img_w, 
            img_h=img_h, 
            lesion_intensity=mass_intensity, 
            lesion_radius=mass_radius, 
            breast_mask=mask,
            oval_mass=np.random.choice([True, False]) # randomly sample oval masses
            )

        if mass is None:
            # no mass was generated
            return None

        img = img + mass
    
    if add_calcs:
        calc = recursively_generate_lesion(
            lesion_type="calc",
            img_w=img_w, 
            img_h=img_h, 
            calc_area_len=calc_area_len, 
            n_calcs=n_calcs, 
            lesion_intensity=calc_intensity, 
            lesion_radius=calc_radius,
            breast_mask=mask
            )
        
        if calc is None:
            # no calc was generated
            return None

        img = img + calc

    img[mask == 0] = 0

    # normalize image
    img = (img-img.min()) / (img.max()-img.min())
    img = img * 255

    # create and save PIL image
    img = (img-img.min()) / (img.max()-img.min())
    img = img * 255
    img = Image.fromarray(np.uint8(img), 'L')
    img.save(os.path.join(save_dir, fname))

    # create and save mask
    mask = np.uint8(mask)
    mask = (mask-mask.min()) / (mask.max()-mask.min())
    mask = mask * 255
    mask = Image.fromarray(np.uint8(mask), 'L')
    mask.save(os.path.join(save_dir, fname.rstrip(".png")+"_mask.png"))

    return per_bckg

def get_noise_sigma(add_noise, sigma_noise):
    
    if add_noise:
        return sigma_noise
    else:
        return None

def create_patches_with_random_params_ds(
    n_imgs, 
    beta_range,
    calc_area_len_range,
    n_calcs_range,
    calc_intensity_range,
    calc_radius_range,
    mass_intensity_range,
    mass_radius_range,
    add_noise,
    sigma_noise,
    img_w, 
    img_h, 
    p_calc,
    p_mass,
    p_breast_border,
    save_dir="",
    out_annots_fname="annots.csv"
    ):

    beta_min, beta_max = beta_range
    calc_area_len_min, calc_area_len_max = calc_area_len_range
    n_calcs_min, n_calcs_max = n_calcs_range
    calc_intensity_min, calc_intensity_max = calc_intensity_range
    calc_radius_min, calc_radius_max = calc_radius_range
    mass_intensity_min, mass_intensity_max = mass_intensity_range
    mass_radius_min, mass_radius_max = mass_radius_range

    annots_df = pd.DataFrame()
    annots_lst = []

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    imgs_dir = os.path.join(save_dir, "images")
    if not os.path.exists(imgs_dir):
        os.mkdir(imgs_dir)

    idx = 0
    while idx < n_imgs-1:
        
        # sample parameters
        #texture
        beta = (beta_max - beta_min) * np.random.rand() + beta_min
        # calc
        n_calcs = np.random.randint(n_calcs_min, n_calcs_max)
        calc_intensity = (calc_intensity_max - calc_intensity_min) * np.random.rand() + calc_intensity_min
        calc_intensity = (calc_intensity_max - calc_intensity_min) * np.random.rand() + calc_intensity_min
        calc_radius = (calc_radius_max - calc_radius_min) * np.random.rand() + calc_radius_min
        calc_area_len = (calc_area_len_max - calc_area_len_min) * np.random.rand() + calc_area_len_min
        # mass
        mass_intensity = (mass_intensity_max - mass_intensity_min) * np.random.rand() + mass_intensity_min
        mass_radius = (mass_radius_max - mass_radius_min) * np.random.rand() + mass_radius_min

        print('Generating image {} of {}'.format(idx, n_imgs))

        _fname='img-{}.png'.format(idx)

        _add_calc = False
        _add_mass = False

        _rand_p = np.random.rand(1)
        if _rand_p < p_calc:
            _add_calc = True
            _class = "calc"
        elif _rand_p < (p_calc+p_mass):
            _add_mass = True
            _class = "mass"
        else:
            _class = "normal"
        
        # not all patches contain borders
        if np.random.rand() < p_breast_border:
            add_borders = True
        else:
            add_borders = False

        per_bckg = synthesize_patch(
                            img_w=img_w,
                            img_h=img_h,
                            beta=beta,
                            add_noise=add_noise,
                            sigma_noise=sigma_noise,
                            add_mass=_add_mass,
                            mass_intensity=mass_intensity,
                            mass_radius=mass_radius,
                            add_calcs=_add_calc,
                            calc_area_len=calc_area_len,
                            n_calcs=n_calcs,
                            calc_intensity=calc_intensity,
                            calc_radius=calc_radius,
                            save_dir=imgs_dir,
                            fname=_fname,
                            add_borders=add_borders
                        )
        
        if per_bckg is None:
            # the patch was not sucessfully generated
            continue

        annots_lst.append(
            {
            "img": os.path.join(imgs_dir, _fname),
            "class": _class,
            "texture_beta": beta,
            "noise_sigma": get_noise_sigma(add_noise, sigma_noise),
            "mass_intensity": mass_intensity,
            "mass_radius": mass_radius,
            "calc_area_len": calc_area_len,
            "n_calcs": n_calcs,
            "calc_intensity": calc_intensity,
            "calc_radius": calc_radius,
            "per_bckg": per_bckg,
            "contain_borders": add_borders
            }
        )

        idx += 1

    annots_df = annots_df.from_dict(annots_lst)
    annots_df.to_csv(os.path.join(save_dir, out_annots_fname))

    # save json with generation specifications
    specs = {
        "mode": "random parameters",
        "n_imgs": n_imgs, 
        "beta": beta_range, 
        "calc_area_len": calc_area_len_range,
        "n_calcs": n_calcs_range,
        "calc_intensity": calc_intensity_range,
        "calc_radius": calc_radius_range,
        "mass_intensity": mass_intensity_range,
        "mass_radius": mass_radius_range,
        "add_noise": add_noise,
        "sigma_noise": sigma_noise,
        "img_w": img_w, 
        "img_h": img_h, 
        "p_calc": p_calc,
        "p_mass": p_mass,
        "p_breast_border": p_breast_border
    }

    with open(os.path.join(save_dir, "specs.json"), 'w', encoding='utf-8') as f:
        json.dump(specs, f, ensure_ascii=False, indent=4)


def main():
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    create_patches_with_random_params_ds(
        n_imgs=args.n_imgs, 
        beta_range=args.beta_range, 
        calc_area_len_range=args.calc_area_len_range,
        n_calcs_range=args.n_calcs_range,
        calc_intensity_range=args.calc_intensity_range,
        calc_radius_range=args.calc_radius_range,
        mass_intensity_range=args.mass_intensity_range,
        mass_radius_range=args.mass_radius_range,
        add_noise=args.add_noise,
        sigma_noise=args.sigma_noise,
        img_w=args.img_w, 
        img_h=args.img_h, 
        p_calc=args.p_calc,
        p_mass=args.p_mass,
        p_breast_border=args.p_breast_border,
        save_dir=args.save_dir,
        out_annots_fname=args.out_annots_fname
    )

if __name__ == "__main__":
    main()