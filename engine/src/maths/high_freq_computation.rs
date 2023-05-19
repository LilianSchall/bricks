use rayon::prelude::*;

fn compute_rows_of_sums(a_row: &[f64], b: &Vec<f64>, k: usize, p: usize) -> Vec<f64> {
    let mut unordered_columns = (0..p)
        .into_par_iter()
        .map(|y| (y, (0..k).map(|x| a_row[x] * b[x * p + y]).sum()))
        .collect::<Vec<(usize, f64)>>();

    unordered_columns.par_sort_by(|left, right| left.0.cmp(&right.0));

    unordered_columns
        .into_iter()
        .map(|(_, col_el)| col_el)
        .collect()
}

pub fn dot_multithreaded(a: &Vec<f64>, b: &Vec<f64>, n: usize, k: usize, p: usize) -> Vec<f64> {
    let mut unordered_rows = (0..n)
        .into_par_iter()
        .map(move |i| {
            let a_row = &a[(i * k)..((i + 1) * k)];
            (i, compute_rows_of_sums(a_row, b, k, p))
        })
        .collect::<Vec<(usize, Vec<f64>)>>();

    unordered_rows.par_sort_by(|left, right| left.0.cmp(&right.0));
    let mut vec: Vec<f64> = vec![];
    unordered_rows.iter_mut().for_each(|(_, i): &mut (usize, Vec<f64>)| vec.append(i));
    vec
}

pub fn dot_monothreaded(a: &Vec<f64>, b: &Vec<f64>, n: usize, k: usize, p: usize) -> Vec<f64> {
    let mut result: Vec<f64> = vec![0.0; n * p];

    for i in 0..n {
        for j in 0..p {
            for l in 0..k {
                result[i * p + j] += a[i * k + l] * b[l * p + j];
            }
        }
    }
    result
}
