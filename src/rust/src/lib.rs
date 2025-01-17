use extendr_api::prelude::*;


use ndarray::Array2;

use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::prelude::*;

mod ort_inference_session;

use rayon::prelude::*;

fn cos_dist(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    let numerator: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let x_norm: f64 = x.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let y_norm: f64 = y.iter().map(|y| y.powi(2)).sum::<f64>().sqrt();

    return 1.0 - numerator / (x_norm * y_norm);
}

#[extendr]
fn multi_cos_distance(a_mat: Robj, b_mat: Robj, indexes: Robj) -> Vec<f64> {
    let a_mat = <ArrayView2<f64>>::try_from(&a_mat).unwrap().to_owned();
    let b_mat = <ArrayView2<f64>>::try_from(&b_mat).unwrap().to_owned();

    let indexes = <ArrayView2<i32>>::try_from(&indexes).unwrap().to_owned();

    indexes
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|x| {
            cos_dist(
                a_mat.row(x[[0]] as usize - 1),
                b_mat.row(x[[1]] as usize - 1),
            )
        })
        .collect()
}

#[extendr]
fn hnsw_join(a_mat: Robj, b_mat: Robj) -> Robj {
    let a_mat = <ArrayView2<f64>>::try_from(&a_mat).unwrap().to_owned();
    let b_mat = <ArrayView2<f64>>::try_from(&b_mat).unwrap().to_owned();

    let a_vec: Vec<Vec<f64>> = a_mat.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();
    let a_borrow_vec: Vec<(&Vec<f64>, usize)> =
        a_vec.iter().enumerate().map(|(a, b)| (b, a)).collect();

    // todo: allow you to change these hyperparameters
    let mut hnsw: Hnsw<f64, DistL2> = Hnsw::new(40, a_mat.nrows(), 8, 20, DistL2);

    hnsw.parallel_insert(&a_borrow_vec);
    hnsw.set_searching_mode(true);

    let b_vec: Vec<Vec<f64>> = b_mat.axis_iter(Axis(0)).map(|row| row.to_vec()).collect();

    let results = hnsw.parallel_search(&b_vec, 20, 30);

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


// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod linkorgsonnx;
    use ort_inference_session;
    fn hnsw_join;
    fn multi_cos_distance;
}
