#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

// basics
use std::error::Error;
use std::path::{Path, PathBuf};
use imageproc::filter;
// arrays/vectors/tensors
use ndarray::{array, Array, Array1, Array2, Array3, Array4, ArrayD, ArrayBase, IxDynImpl};
use ndarray::{s, Axis, Dim, Ix2, IxDyn};
use ndarray::{ViewRepr, OwnedRepr};
// images
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba, };
use image::imageops::FilterType;
// use imageproc::drawing::draw_filled_rect_mut;
// use imageproc::rect::Rect;
// machine learning
use ort::{GraphOptimizationLevel, ModelMetadata, Session};
// graphics
use tiny_skia::*;

use ndarray_stats::QuantileExt;

use serde_json::{Value, Map};
use std::collections::HashMap;


// mod utils2;
use crate::utils2::{image_to_onnx_input, crop_square};



fn center_coords_to_box_coords(cx: i32, cy: i32, w: i32, h: i32) -> (i32, i32, i32, i32) {
    let x_left = cx - w / 2;
    let x_right = cx + w / 2;
    let y_top = cy - h / 2;
    let y_bottom = cy + h / 2;

    return (x_left, x_right, y_top, y_bottom);
}
fn center_coords_to_box_coords_f32(cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    let x_left = cx - w / 2.;
    let x_right = cx + w / 2.;
    let y_top = cy - h / 2.;
    let y_bottom = cy + h / 2.;

    return (x_left, x_right, y_top, y_bottom);
}


// fn thresholded_argmax(a: Array<f32, Dim<IxDynImpl>>, threshold: f32) -> Array1<u32> { // takes argmax of rows above threshold // https://stackoverflow.com/a/57963733/15275714
// fn thresholded_argmax(a: Array<f32, IxDyn>, threshold: f32) -> Array1<u32> { // takes argmax of rows above threshold // https://stackoverflow.com/a/57963733/15275714
fn thresholded_argmax_2d(a: Array<f32, Ix2>, threshold: f32) -> Array1<u32> { // takes argmax of rows above threshold // https://stackoverflow.com/a/57963733/15275714
    let mut indices = Array1::zeros(a.shape()[0]);
    for (i, row) in a.axis_iter(Axis(0)).enumerate() {
        let (max_idx, max_val) =
            row.iter()
                .enumerate()
                .fold((0, row[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });
        // println!("max idx for row {}: {}", i, max_idx); // DEBUG
        if max_val > threshold { indices[i] = max_idx as u32; }
        else { indices[i] = 0; }
    }
    return indices;
}

fn argmax(a: Array1<f32>) -> usize {
    let (max_idx, max_val) = a.iter().enumerate().fold((0 as usize, 0.), |(max_idx, max_val), (cur_idx, &cur_val)| if max_val < cur_val { (cur_idx, cur_val) } else { (max_idx, max_val) } );
    return max_idx;
}

fn thresholded_mask(a: Array<f32, Ix2>, threshold: f32) -> Array1<bool> { // takes argmax of rows above threshold // https://stackoverflow.com/a/57963733/15275714
    let mut indices = Array1::from_elem(a.shape()[0], false);
    for (i, row) in a.axis_iter(Axis(0)).enumerate() {
        let (max_idx, max_val) =
            row.iter()
                .enumerate()
                .fold((0, row[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });
            // println!("max idx for row {}: {}", i, max_idx); // DEBUG
            if max_val > threshold { indices[i] = true; }
            else { indices[i] = false; }
    }
    return indices;
}

// TODO study how this code works.
// fn apply_mask<T>(array: &Array2<T>, mask: &Array1<bool>) -> Array2<T> // ALT1-GPT
// where
//     T: Clone,
// {
//     // Filter the rows of the array based on the mask
//     let filtered_rows: Vec<_> = array.axis_iter(Axis(0))
//                                       .zip(mask.iter())
//                                       .filter_map(|(row, &m)| if m { Some(row.to_owned()) } else { None })
//                                       .collect();

//     // Stack the filtered rows to create the resulting 2D array
//     // We use `stack` to rebuild the 2D array from filtered rows, handling the empty case correctly
//     let result = if filtered_rows.is_empty() {
//         Array2::zeros((0, array.ncols())) // or handle as per the appropriate dimensionality
//     } else {
//         ndarray::stack(Axis(0), &filtered_rows).unwrap() // Unwrap is safe here given the non-empty case
//     };

//     result
// }



fn apply_mask<T>(array: &Array2<T>, mask: &Array1<bool>) -> Array2<T> // ALT2-GPT
where
    T: Clone,
{
    // let length = count_trues(&mask.into_dyn());



    // Collect rows where the mask is true
    let filtered_rows: Vec<_> = array.axis_iter(Axis(0))
        .enumerate()
        .filter(|(i, _)| mask[*i])
        // .map(|(_, row)| row.to_owned())
        .map(|(_, row)| row)//.to_owned())
        .collect();

    // let filtered_rows_array = vec_to_arr(filtered_rows);

    // let filtered_rows_array = 
    // let filtered_rows_array = vec_of_arrays_to_array2(filtered_rows)?; // 2D array 이게 아니라, 1d array의 rust native array를 원하는거였군.

    


    // Reconstruct the 2D array from the filtered rows // ALT1
    return ndarray::stack(Axis(0), &filtered_rows).unwrap();
    // ndarray::stack(Axis(0), &filtered_rows_array).unwrap()
    // .unwrap_or_else(|_| Array2::default((0, array.ncols())))

    // let array_2d = ndarray::stack(Axis(0), &filtered_rows.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap(); // ALT2
}

fn vec_to_arr<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}

fn count_trues(array: &ArrayD<bool>) -> usize {
    array.iter().filter(|&&elem| elem).count()
}

fn vec_of_arrays_to_array2<T>(vec: Vec<Array1<T>>) -> Result<Array2<T>, ndarray::ShapeError>
where
    T: Clone,
{
    // Use `stack` to combine the array of 1D arrays into a 2D array
    // Axis(0) means stacking along the rows, so each 1D array becomes a row in the 2D array
    ndarray::stack(Axis(0), &vec.iter().map(|a| a.view()).collect::<Vec<_>>())
}

pub struct RtDetr {
    file: PathBuf,
    model: Session,
    metadata: ModelMetadata,
    size: u32,
    threshold: f32,
}


impl RtDetr {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        // let file = PathBuf::from("./assets/onnx/rt-detr/rtdetr-l.onnx"); // ALT1
        let file = PathBuf::from("./assets/onnx/rt-detr/rtdetr-l_json.onnx"); // ALT1.1
        // let file = PathBuf::from("./assets/onnx/yolov9c.onnx"); // ALT2
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .with_model_from_file(&file)?;
        let metadata = model.metadata()?;
        let size = 640; 
        let threshold = 0.5;
        
        // println!{"model metadata: {}", metadata};

        Ok( Self { file, model, metadata, size, threshold } )
    }

    pub fn process(self) -> Result<(), Box<dyn Error>> {
        // load image
        let mut original_image = ImageReader::open("./test_images/obj_det/computer_desk.jpg")?.decode()?;
        // let mut original_image = ImageReader::open("./test_images/obj_det/women_umbrella.jpg")?.decode()?;
        let original_image_size = original_image.dimensions();
        
        // resize image
        let mut image = crop_square(original_image); // ALT1
        // let mut image = pad_square(original_image); // ALT2
        image = image.resize_exact(self.size, self.size, FilterType::CatmullRom);
        let image_size = image.dimensions();
        
        // convert image into input
        let image_array = image_to_onnx_input(image.clone());
        let ort_inputs = ort::inputs!["images" => image_array.view()]?; 
        
        // run it thru model
        let ort_outputs = self.model.run(ort_inputs)?; // NOTE if image size is not right, this crashes silently (without panicking).
        let outputs = ort_outputs["output0"].extract_tensor::<f32>()?;
        let batch_outputs_view = outputs.view().clone().into_owned();
        
        let outputs_view = batch_outputs_view.slice(s![0,..,..]).view().clone().into_owned(); // choose only the first image (temp)
        
        // post-process predictions
        // println!("{:?}", outputs_view);
        // println!("{:?}", output0_view.slice(s![0, 0, ..]));

        // label
        let class_dict_raw = self.metadata.custom("names_json").unwrap().unwrap();
        let class_dict: Value = serde_json::from_str(&class_dict_raw)?;
        let class_list_raw = self.metadata.custom("names_json_array").unwrap().unwrap();
        let class_list: Value = serde_json::from_str(&class_list_raw)?; // not bad!
        // println!("{:?}", class_list);

        // process preds into boxes: filter confidences, get coordinates, run NMS
        // NOTE still want to understand how it works (although this isn't strictly relevant/important): read YOLACT's paper and see what it says, and ask ChatGPT. 

        // filter confidences
        // NOT not sure about the code quality / effectiveness here. perhaps refactor depending on what code reviews/GPT/readability/efficiency/reusability says on it.
        // let confident = output0_view.slice(s![..,4..]).max(Axis(1)) > 0.9;
        let confident = thresholded_mask(outputs_view.slice(s![..,4..]).clone().into_owned(), self.threshold);
        let confident_indices = thresholded_argmax_2d(outputs_view.slice(s![..,4..]).clone().into_owned(), self.threshold);
        // println!("{:?}", confident);
        
        let confident_outputs_view = apply_mask(&outputs_view, &confident);
        println!("{:?}", confident_outputs_view);

        let relevant_outputs: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = apply_mask(&outputs_view, &confident);

        let mut path_builder = PathBuilder::new();

        for (i, box_pred) in relevant_outputs.axis_iter(Axis(0)).enumerate() {

            let (r_cx, r_cy, r_w, r_h) = (box_pred[0], box_pred[1], box_pred[2], box_pred[3]); // relative
            // u32
            // let (i_cx, i_cy, i_w, i_h) = ((r_cx * image_size.0 as f32) as u32, (r_cy * image_size.1 as f32) as u32, (r_w * image_size.0 as f32) as u32, (r_h * image_size.1 as f32) as u32); // input image coordinates
            // let (o_cx, o_cy, o_w, o_h) = ((r_cx * original_image_size.0 as f32) as u32, (r_cy * original_image_size.1 as f32) as u32, (r_w * original_image_size.0 as f32) as u32, (r_h * original_image_size.1 as f32) as u32); // original image coordinates
            // i32
            // let (i_cx, i_cy, i_w, i_h) = ((r_cx * image_size.0 as f32) as i32, (r_cy * image_size.1 as f32) as i32, (r_w * image_size.0 as f32) as i32, (r_h * image_size.1 as f32) as i32); // input image coordinates
            // let (o_cx, o_cy, o_w, o_h) = ((r_cx * original_image_size.0 as f32) as i32, (r_cy * original_image_size.1 as f32) as i32, (r_w * original_image_size.0 as f32) as i32, (r_h * original_image_size.1 as f32) as i32); // original image coordinates
            // f32
            let (i_cx, i_cy, i_w, i_h) = (r_cx * image_size.0 as f32, r_cy * image_size.1 as f32, r_w * image_size.0 as f32, r_h * image_size.1 as f32); // input image coordinates
            let (o_cx, o_cy, o_w, o_h) = (r_cx * original_image_size.0 as f32, r_cy * original_image_size.1 as f32, r_w * original_image_size.0 as f32, r_h * original_image_size.1 as f32); // original image coordinates
            // TODO readability.

            let (i_x_l, i_x_r, i_y_t, i_y_b) = center_coords_to_box_coords_f32(i_cx, i_cy, i_w, i_h);
            
            let label = class_list[argmax(box_pred.slice(s![4..]).clone().into_owned())].as_str().unwrap();
            println!("{:?}", (label)); // DEBUG; EVAL

            path_builder.push_rect(Rect::from_ltrb(i_x_l , i_y_t, i_x_r, i_y_b).unwrap());

        }

        // println!("{:?}", ()); // DEBUG; EVAL
        

        // for box_pred in output_view.axis_iter(Axis(0)) {
        //     println!("box_pred: {:?}", box_pred); // DEBUG
        //     let indices = thresholded_argmax_2d(box_pred.into_owned(), 0.90);
        //     let nonzero_indices = drop_zeros(indices);
        //     // println!("box_pred (plus 1): {:?}", nonzero_indices.clone() + 1); // DEBUG

        //     let mut chars: Vec<String> = Vec::new();
        //     // convert indices into characters
        //     // for idx in indices.axis_iter(Axis(0)) {
        //     for idx in nonzero_indices.iter() {
        //         let char = self.char_list.get(*idx as usize).unwrap();
        //         // println!("char: {}", char);
        //         chars.push(char.to_string());
        //     }
        //     let text: String = chars.join("");
        //     println!("detected text: {}", text);
        // }
        
        // create masks & visualize
        
        // skia: rectangle, color, text(?! may not support. bitmap texting is fine, as long as safe)

        let img_rgba = image.clone().to_rgba8(); 
        let mut pixmap = Pixmap::from_vec( // NOTE compiler warns
          img_rgba.into_raw(), // This converts the DynamicImage into a Vec<u8> of raw pixels
          IntSize::from_wh(image_size.0 as u32, image_size.1 as u32).unwrap()
        ).expect("Failed to create Pixmap");

        let paint = Paint {
            shader: Shader::SolidColor(Color::from_rgba8(180, 229, 255, 255)), // Green color
            anti_alias: true,
            ..Default::default()
          };

        let path = path_builder.finish().unwrap();

        // pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
        let stroke = Stroke::default();
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);

        pixmap.save_png("viz.png").unwrap();

        // process preds into masks: matrix manipulation (transpose, flatten), multiply by weights, reshape/un-flatten, threshold, convert to binary, crop, combine layers, create translucent mask visualizations

        // DEBUG
        image.save("inspect.png")?;



        Ok(())
    }
}

