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

mod yolo;
mod utils2;

struct Detr {
    file: PathBuf,
    model: Session,
    // size: u32,
}

impl Detr {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let file = PathBuf::from("./assets/onnx/yolov8n.onnx");
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(&file)?;
        // let size = 224; // NOTE

        Ok( Self { file, model, } ) // size } )
    }

    pub fn process(self) -> Result<(), Box<dyn Error>> {
        // load image
        let mut det_image = ImageReader::open("./test_images/sparse_text.jpg")?.decode()?;

        // convert image into input


        // run it thru model


        // post-process predictions

        
        // create masks & visualize


        Ok( () )
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // println!("Hello, world!");

    let yolo_model = yolo::Yolo::new()?;
    yolo_model.process();


    Ok(())
}
