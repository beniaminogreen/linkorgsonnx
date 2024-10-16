use extendr_api::prelude::*;

use std::path::Path;

use ndarray::Array2;
use ort::{GraphOptimizationLevel, Session, ValueType, CUDAExecutionProvider};
use tokenizers::Tokenizer;

use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::prelude::*;

use rayon::prelude::*;

fn cos_dist(x: ArrayView1<f64>, y : ArrayView1<f64> ) -> f64{
    let numerator : f64 =  x.iter().zip(y.iter()).map(|(a,b)| a*b).sum();
    let x_norm : f64 = x.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let y_norm : f64 = y.iter().map(|y| y.powi(2)).sum::<f64>().sqrt();

    return 1.0-numerator / (x_norm * y_norm)
}

#[extendr]
fn multi_cos_distance(a_mat : Robj, b_mat : Robj, indexes : Robj) -> Vec<f64>{
    let a_mat = <ArrayView2<f64>>::try_from(&a_mat).unwrap().to_owned();
    let b_mat = <ArrayView2<f64>>::try_from(&b_mat).unwrap().to_owned();

    let indexes = <ArrayView2<i32>>::try_from(&indexes).unwrap().to_owned();

    indexes.axis_iter(Axis(0))
        .into_par_iter()
        .map(|x| cos_dist(a_mat.row(x[[0]] as usize -1 ), b_mat.row(x[[1]] as usize -1 ))).collect()
}

#[extendr]
fn hnsw_join(a_mat: Robj, b_mat: Robj) -> Robj {
    let a_mat = <ArrayView2<f64>>::try_from(&a_mat).unwrap().to_owned();
    let b_mat = <ArrayView2<f64>>::try_from(&b_mat).unwrap().to_owned();

    let a_vec: Vec<Vec<f64>> = a_mat.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();
    let a_borrow_vec : Vec<(&Vec<f64>, usize)> = a_vec.iter().enumerate().map(|(a, b)| (b, a)).collect();


    let mut hnsw : Hnsw<f64, DistL2> = Hnsw::new(40, a_mat.nrows(), 8, 20, DistL2);

    hnsw.parallel_insert(&a_borrow_vec);
    hnsw.set_searching_mode(true);

    let b_vec: Vec<Vec<f64>> = b_mat.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();

    let results = hnsw.parallel_search(&b_vec,20, 30);

    let mut pairs = Vec::new();
    for (i, neighbors) in results.iter().enumerate() {
        for neighbor in neighbors {
            pairs.push((i, neighbor.d_id));
        }
    }

    let mut out_arr: Array2<i32> = Array2::zeros((pairs.len(), 2));

    for (idx, (i, j)) in pairs.into_iter().enumerate() {
        out_arr[[idx, 0]] = j as i32 + 1;
        out_arr[[idx, 1]] = i as i32 + 1;
    }

    Robj::try_from(&out_arr).into()
}

#[extendr]
struct ORTSession {
    session: Session,
    tokenizer: Tokenizer,
    embedding_dim: usize,
    needs_token_types: bool
}

#[extendr]
impl ORTSession {
    fn new_from_path(path: &str, tokenizer_path: &str) -> Self {
        let session = Session::builder()
            .expect("Could not build session")
            .with_execution_providers([CUDAExecutionProvider::default().build()]) // , CPUExecutionProvider::default().build()])
            .expect("Could not set execution provider priority")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("could not set optimization level")
            .commit_from_file(path)
            .expect("could not commit from path");

        let output_value = &session.outputs[0].output_type;

        let embedding_dim: usize = match output_value {
            ValueType::Tensor { dimensions, .. } => dimensions[2] as usize,
            _ => panic!("Could not get"),
        };

        let tokenizer = Tokenizer::from_file(Path::new(tokenizer_path)).unwrap();

        let needs_token_types = session.inputs.len() == 3;

        Self {
            session,
            tokenizer,
            embedding_dim: embedding_dim,
            needs_token_types,
        }
    }

    fn get_embedding(&self, input : &str) -> Vec<i64> {
            let encoding = self.tokenizer.encode(input, true).unwrap();

            let ids: Vec<i64> = encoding.get_ids().iter().map(|x| *x as i64).collect();

            ids
    }

    fn run_model(&self, inputs: Strings, output_index : usize, mean_pooling_needed : bool) -> Robj {
        let mut out_arr: Array2<f32> = Array2::zeros((inputs.len(), self.embedding_dim));

        // for (index_chunk, input_chunk) in inputs.iter().enumerate().windows(1000) {
        // }

        let chunk_size =   1000;
        for (chunk_index, input_chunk) in inputs.chunks(chunk_size).enumerate() {
            let input_chunk: Vec<String> = input_chunk
                .iter()
                .map(|s| s.to_string()) // Convert each R string to Rust String
                .collect();

            let input_len = input_chunk.len();

            let encodings = self.tokenizer.encode_batch(input_chunk, true).unwrap();

            // Get the padded length of each encoding.
            let padded_token_length = encodings
                .iter()
                .map(|x| x.len())
                .max()
                .expect("could not find input length");

            // Get our token IDs & mask as a flattened array.
            let ids: Vec<i64> = encodings
                .iter()
                .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
                .collect();
            let mask: Vec<i64> = encodings
                .iter()
                .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
                .collect();
            let type_ids: Vec<i64> = encodings
                .iter()
                .flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
                .collect();

            // Convert our flattened arrays into 2-dimensional tensors of shape [N, L].

            let a_ids = Array2::from_shape_vec([input_len, padded_token_length], ids).unwrap();
            let a_mask = Array2::from_shape_vec([input_len, padded_token_length], mask).unwrap();
            let a_type_ids = Array2::from_shape_vec([input_len, padded_token_length], type_ids).unwrap();

            let (batch_size, seq_len) = a_mask.dim();
            let hidden_size = self.embedding_dim;

            let attention_mask_expanded = a_mask
                .clone()
                .insert_axis(Axis(2)) // Add a new axis
                .broadcast((batch_size, seq_len, hidden_size)) // Broadcast to the shape of token_embeddings
                .unwrap()
                .mapv(|x| x as f32);

            let outputs;
            if self.needs_token_types {
                outputs = self
                    .session
                    .run(ort::inputs![a_ids, a_mask, a_type_ids].unwrap())
                    .unwrap();
            } else {
                outputs = self
                    .session
                    .run(ort::inputs![a_ids, a_mask].unwrap())
                    .unwrap();
            }

            let embeddings;
            if mean_pooling_needed {
                embeddings = outputs[output_index]
                    .try_extract_tensor::<f32>()
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap()
                    .to_owned();
            } else {
                let token_embeddings = outputs[output_index]
                    .try_extract_tensor::<f32>()
                    .unwrap()
                    .into_dimensionality::<Ix3>()
                    .unwrap()
                    .to_owned();

                let weighted_sum = token_embeddings * &attention_mask_expanded;
                let sum_embeddings = weighted_sum.sum_axis(Axis(1));

                // Compute the sum of the attention mask
                let sum_mask = attention_mask_expanded
                    .sum_axis(Axis(1))
                    .mapv(|x| x.max(1e-9)); // Ensure no zero division

                // Compute the mean by dividing the sum of token embeddings by the sum of the attention mask
                embeddings = &sum_embeddings / &sum_mask;
            }

            let start_row = chunk_size * chunk_index;
            let end_row = start_row + input_len;

            let mut slice = out_arr.slice_mut(s![start_row..end_row, ..]);
            slice.assign(&embeddings);
        }

        out_arr.try_into().unwrap()
    }
}

#[extendr]
// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod linkorgsonnx;
    impl ORTSession;
    fn hnsw_join;
    fn multi_cos_distance;
}
