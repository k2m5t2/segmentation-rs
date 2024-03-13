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
use ort::{GraphOptimizationLevel, ModelMetadata, Session};

use serde_json::{Value, Map};
use std::collections::HashMap;

// mod utils2;
use crate::utils2::{image_to_onnx_input, crop_square};


pub struct Yolo {
    file: PathBuf,
    model: Session,
    metadata: ModelMetadata,
    size: u32,
}

impl Yolo {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        // let file = PathBuf::from("./assets/onnx/yolov8m-seg.onnx"); // ALT1
        let file = PathBuf::from("./assets/onnx/yolov8m-seg_json.onnx"); // ALT1.1
        // let file = PathBuf::from("./assets/onnx/yolov9c.onnx"); // ALT2
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .with_model_from_file(&file)?;
        let size = 640; 
        
        let metadata = model.metadata()?;
        // println!{"model metadata: {}", metadata};

        Ok( Self { file, model, metadata, size } )
    }

    pub fn process(self) -> Result<(), Box<dyn Error>> {
        // load image
        let mut original_image = ImageReader::open("./test_images/obj_det/computer_desk.jpg")?.decode()?;
        
        // resize image
        let mut image = crop_square(original_image); // ALT1
        // let mut image = pad_square(original_image); // ALT2
        image = image.resize_exact(self.size, self.size, FilterType::CatmullRom);
        
        // convert image into input
        let image_array = image_to_onnx_input(image);
        let ort_inputs = ort::inputs!["images" => image_array.view()]?; 
        
        // run it thru model
        let ort_outputs = self.model.run(ort_inputs)?; // NOTE if image size is not right, this crashes silently (without panicking).
        let output0 = ort_outputs["output0"].extract_tensor::<f32>()?;
        let output0_view = output0.view().clone().into_owned();
        let output1 = ort_outputs["output1"].extract_tensor::<f32>()?;
        let output1_view = output1.view().clone().into_owned();
        
        // post-process predictions
        println!("{:?}", output0_view);
        // println!("{:?}", output0_view.slice(s![0, 0, ..]));

        // label
        let class_dict_raw = self.metadata.custom("names_json").unwrap().unwrap();
        let class_dict: Value = serde_json::from_str(&class_dict_raw)?;
        let class_list_raw = self.metadata.custom("names_json_array").unwrap().unwrap();
        let class_list: Value = serde_json::from_str(&class_list_raw)?; // not bad!
        // println!("{:?}", class_list);

        // process preds into boxes: filter confidences, get coordinates, run NMS
        // still want to understand how it works (although this isn't strictly relevant/important): read YOLACT's paper and see what it says, and ask ChatGPT. 

        
        // create masks & visualize
        
        // skia: rectangle, color, text(?! may not support. bitmap texting is fine, as long as safe)

        // process preds into masks: matrix manipulation (transpose, flatten), multiply by weights, reshape/un-flatten, threshold, convert to binary, crop, combine layers, create translucent mask visualizations





        Ok(())
    }
}

