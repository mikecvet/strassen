use clap::{arg, Command};
use rand::{Rng, thread_rng};
use std::thread;
use strassen::{matrix::Matrix, timer::Timer, mult::mult_naive, mult::mult_strassen, mult::mult_transpose};
use rayon::prelude::*;

fn
record_trial (a: &Matrix, b: &Matrix, 
              timer: &mut Timer, 
              multiplier: fn(&Matrix, &Matrix) -> Matrix) -> u128 {

    timer.start();
    let c = a.mult(&b, multiplier);
    let duration = timer.stop();

    assert!(c.rows == a.rows);
    assert!(c.cols == a.rows);

    return duration;
}

fn 
time_multiplication (lower: usize, upper: usize, factor: usize, trials: usize) {
    let mut rng = thread_rng();
    let mut skip_naive = false;

    println!("running {} groups of {} trials with bounds between [{}->{}, {}->{}]", factor, trials, lower, lower * factor, upper, upper * factor);

    let mut timer = Timer::new();

    println!("x y nxn naive transpose strassen");

    for i in 1..(factor + 1) {
        let x: usize = i * lower;
        let y: usize = i * upper;

        let mut naive_accumulator:u128 = 0;
        let mut transpose_accumulator:u128 = 0;
        let mut strassen_accumulator:u128 = 0;

        let mut v1:Vec<f64> = Vec::with_capacity((x * y) as usize);
        let mut v2:Vec<f64> = Vec::with_capacity((x * y) as usize);

        for _ in 0..(x * y) {
            v1.push(rng.gen_range(0.0..1000000.0));
            v2.push(rng.gen_range(0.0..1000000.0));
        }

        let a = Matrix::with_array(v1, x, y);
        let b = Matrix::with_array(v2, y, x);

        // Run the timed tests
        for _ in 0..trials {

            if !skip_naive {
                naive_accumulator += record_trial(&a, &b, &mut timer, mult_naive);
            }

            transpose_accumulator += record_trial(&a, &b, &mut timer, mult_transpose);
            strassen_accumulator += record_trial(&a, &b, &mut timer, mult_strassen);
        }

        let d = trials as f64;
        let naive_time = (naive_accumulator as f64) / d;

        println!("{} {} {} {:.2} {:.2} {:.2}", x, y, x * y,
          (naive_accumulator as f64) / d, (transpose_accumulator as f64) / d,
            (strassen_accumulator as f64) / d);

        //if naive_time > 90000.0 {
        //    skip_naive = true;
       // }
    }
}

fn 
main () {
    let matches = Command::new("strassen")
    .version("0.1")
    .about("Evaluating different matrix multiplication algorithms")
    .arg(arg!(--upper <VALUE>).required(false))
    .arg(arg!(--lower <VALUE>).required(false))
    .arg(arg!(--factor <VALUE>).required(false))
    .arg(arg!(--trials <VALUE>).required(false))
    .get_matches();

    let upper_opt = matches.get_one::<String>("upper");
    let lower_opt = matches.get_one::<String>("lower");
    let factor_opt = matches.get_one::<String>("factor");
    let trials_opt = matches.get_one::<String>("trials");

    let default_lower:usize = 35;
    let default_upper:usize = 400;
    let default_factor:usize = 2;
    let default_trials:usize = 100;

    let (lower, upper) = match (lower_opt, upper_opt) {
        (Some(lower_str), Some(upper_str)) => {
            let lower:usize = lower_str.parse().unwrap();
            let upper:usize = upper_str.parse().unwrap();
            (lower, upper)
        },
        (Some(lower_str), None) => {
            let lower:usize = lower_str.parse().unwrap();
            (lower, default_upper)
        },
        (None, Some(upper_str)) => {
            let upper:usize = upper_str.parse().unwrap();
            (default_lower, upper)
        },
        (None, None) => {
            (default_lower, default_upper)
        }
    };

    let factor = match factor_opt {
        Some(factor_str) => factor_str.parse().unwrap(),
        _ => default_factor
    };

    let trials = match trials_opt {
        Some(trials_str) => trials_str.parse().unwrap(),
        _ => default_trials
    };

    time_multiplication(lower, upper, factor, trials);
}
