#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

// basics
use std::error::Error;
use std::path::{Path, PathBuf};
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

// mod utils2;
use crate::utils2::image_to_onnx_input;


pub struct Yolo {
    file: PathBuf,
    model: Session,
    size: u32,
}

impl Yolo {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let file = PathBuf::from("./assets/detr.onnx");
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .with_model_from_file(&file)?;
        let size = 640; 

        Ok( Self { file, model, size } )
    }

    pub fn process(self) -> Result<(), Box<dyn Error>> {
        // load image
        let mut det_image = ImageReader::open("./test_images/obj_det/computer_desk.jpg")?.decode()?;
        
        det_image = det_image.resize_exact(self.size, self.size, FilterType::CatmullRom);

        // convert image into input
        let det_image_array = image_to_onnx_input(det_image);
        let det_inputs = ort::inputs!["images" => det_image_array.view()]?; 

        // run it thru model



        // post-process predictions

        
        // create masks & visualize

        Ok(())
    }
}

