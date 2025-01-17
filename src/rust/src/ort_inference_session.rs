use extendr_api::prelude::*;

use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session, ValueType};
use tokenizers::Tokenizer;

use std::path::Path;

#[extendr]
struct ORTSession {
    session: Session,
    tokenizer: Tokenizer,
    embedding_dim: usize,
    needs_token_types: bool,
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
            _ => panic!("Could not get the embedding dimension"),
        };

        let tokenizer = Tokenizer::from_file(Path::new(tokenizer_path)).unwrap();

        // if the
        let needs_token_types = session.inputs.len() == 3;

        Self {
            session,
            tokenizer,
            embedding_dim,
            needs_token_types
        }
    }

    fn run_model(&self, inputs: Strings, output_index: usize, mean_pooling_needed: bool) -> Robj {
        let mut out_arr: Array2<f32> = Array2::zeros((inputs.len(), self.embedding_dim));

        // iterate over the input in chunks of 1000 to keep memory usage low
        let chunk_size = 1000;
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
                .flat_map(|e| e.get_ids().into_iter().map(|i| *i as i64))
                .collect();
            let mask: Vec<i64> = encodings
                .iter()
                .flat_map(|e| e.get_attention_mask().into_iter().map(|i| *i as i64))
                .collect();

            // Convert our flattened arrays into 2-dimensional tensors of shape [N, L].

            let a_ids = Array2::from_shape_vec([input_len, padded_token_length], ids).unwrap();
            let a_mask = Array2::from_shape_vec([input_len, padded_token_length], mask).unwrap();

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
                let type_ids: Vec<i64> = encodings
                    .iter()
                    .flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
                    .collect();
                let a_type_ids =
                    Array2::from_shape_vec([input_len, padded_token_length], type_ids).unwrap();

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

extendr_module! {
    mod ort_inference_session;
    impl ORTSession;
}
