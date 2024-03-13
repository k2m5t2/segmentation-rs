// basics
use std::error::Error;
use std::path::{Path, PathBuf};
use std::cmp::max;
// arrays/vectors/tensors
use ndarray::{array, Array, Array1, Array2, Array3, Array4, ArrayBase, IxDynImpl};
use ndarray::{s, Axis, Dim, IxDyn};
use ndarray::{ViewRepr, OwnedRepr};
// images
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba, };
use image::imageops::FilterType;
// use imageproc::drawing::draw_filled_rect_mut;
// use imageproc::rect::Rect;
// machine learning
use ort::{Session, GraphOptimizationLevel};

pub fn image_to_onnx_input(image: DynamicImage) -> Array4<f32> {
    let mut img_arr = image.to_rgb8().into_vec();
    let (width, height) = image.dimensions();
    let channels = 3;
    let mut onnx_input = Array::zeros((1, channels, height as _, width as _));
    for (x, y, pixel) in image.into_rgb8().enumerate_pixels() {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        // Set the RGB values in the array
        onnx_input[[0, 0, y as _, x as _]] = (r as f32) / 255.;
        onnx_input[[0, 1, y as _, x as _]] = (g as f32) / 255.;
        onnx_input[[0, 2, y as _, x as _]] = (b as f32) / 255.;
      };
    onnx_input
    //   x_d = np.array(img_d).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)/256 // HWC -> NCHW
  }

// pub fn pad_square(image: DynamicImage) -> DynamicImage {
//     let (width, height) = image.dimensions();
//     let max_dim = max(width, height);

//     // let mut padded_image = ImageBuffer::new(max_dim, max_dim);
//     let enlarged = image.resize_exact(, nheight, filter)
    
// }

pub fn crop_square(image: DynamicImage) -> DynamicImage {
  let (width, height) = image.dimensions();

  // Determine starting point and size
  let (x, y, size) = if width > height {
      ((width - height) / 2, 0, height)
  } else {
      (0, (height - width) / 2, width)
  };

  // Crop the image to square
  image.clone().crop(x, y, size, size)
}