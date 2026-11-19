# Example image

`chelsea_224.png` is a 224 x 224 crop of the "chelsea" test image shipped with
scikit-image (`skimage/data/chelsea.png`, 451 x 300 RGB, photograph of Chelsea the cat).

- Licence: CC0 / no copyright restrictions ("No copyright restrictions. CC0 by the
  photographer (Stefan van der Walt)", scikit-image `data.chelsea` docstring).
- Crop: plain crop, no resampling, PIL box `(left, top, right, bottom) = (128, 48, 352, 272)`
  on the original image, i.e. rows 48..271 and columns 128..351. Both offsets are
  multiples of 16, so the 16 x 16 block grid is aligned with the original image.
