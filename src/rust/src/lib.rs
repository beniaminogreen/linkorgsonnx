use extendr_api::prelude::*;

use std::{
    path::Path,
};

use ndarray::{array, concatenate, s, Array1, ArrayViewD, ArrayView2, Axis, Array, Array2, Ix2};
use ort::{inputs, GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

fn character_tokenize(strings : Strings, max_input_len : usize) -> Array2<i32> {

    let mut output = Array2::<i32>::zeros((strings.len(), max_input_len));

    for (i, string) in strings.iter().enumerate() {
        for (j, x) in string.chars().enumerate().take(max_input_len) {
            let token: u64 = x.into();
            output[[i, j]] = token as i32 -32 as i32;
        }
    }

    output
}

#[extendr]
struct ORTSession {
    session: Session
}

#[extendr]
impl ORTSession {
    fn new_from_url(num_threads: usize, url: &str) -> Self {
        let session = Session::builder()
            .expect("Could not build session")
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .expect("could not set optimization level")
            .with_intra_threads(num_threads)
            .expect("could not set # of threads")
            .commit_from_url(url)
            .expect("could not commit from url");

        Self { session }
    }

    fn new_from_path(num_threads: usize, path: &str) -> Self {
        let session = Session::builder()
            .expect("Could not build session")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("could not set optimization level")
            .with_intra_threads(num_threads)
            .expect("could not set # of threads")
            .commit_from_file(path)
            .expect("could not commit from path");

        Self { session }
    }

    fn run_model(&self, x: Strings){
        let x = character_tokenize(x, 40);

        let x = x.row(0);

        let outputs = self.session.run(inputs![x].unwrap()).unwrap();

        dbg!(outputs);
    }
}


// #[extendr]
// fn tokenize(prompt : &str) {
//         // let tokenizer = Tokenizer::from_file(Path::new("/home/beniamino/programming/Rort/tokenizer.json")).unwrap();

//         let tokens = tokenizer.encode(prompt, false).unwrap();
//         let tokens = tokens
//             .get_ids()
//             .iter()
//             .map(|i| *i as i64)
//             .collect::<Vec<_>>();

//         let tokens = Array1::from_iter(tokens.iter().cloned());

//         dbg!(tokens);
// }

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod linkorgsonnx;
    // fn character_tokenize;
    impl ORTSession;
}
