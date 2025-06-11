use serde_json;
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::time::Instant;
use tch::{CModule, Device, Kind, Tensor};

// Constants
const DEFAULT_WORK_DIR: &str = "work";
const DEFAULT_TEST_DATA: &str = "test/input.txt";
const DEFAULT_TEST_OUTPUT: &str = "pred.txt";
const CONTEXT_LENGTH: i64 = 32;

// Struct to hold command line arguments
struct Args {
    work_dir: String,
    test_data: String,
    test_output: String,
    time: bool,
    torchscript: bool,
}

// Parse command line arguments
fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();
    let mut work_dir = String::from(DEFAULT_WORK_DIR);
    let mut test_data = String::from(DEFAULT_TEST_DATA);
    let mut test_output = String::from(DEFAULT_TEST_OUTPUT);
    let mut time = false;
    let mut torchscript = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--work_dir" => {
                if i + 1 < args.len() {
                    work_dir = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--test_data" => {
                if i + 1 < args.len() {
                    test_data = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--test_output" => {
                if i + 1 < args.len() {
                    test_output = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--time" => {
                time = true;
                i += 1;
            }
            "--torchscript" => {
                torchscript = true;
                i += 1;
            }
            _ => {
                println!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    Args {
        work_dir,
        test_data,
        test_output,
        time,
        torchscript,
    }
}

// Load vocabulary from JSON file
fn load_vocab(vocab_path: &str) -> io::Result<(HashMap<String, i64>, Vec<String>)> {
    println!("Loading vocabulary from {}", vocab_path);
    let start = Instant::now();

    let file = File::open(vocab_path)?;
    let reader = BufReader::new(file);

    // Parse JSON into HashMap
    let vocab_map: HashMap<String, i64> = serde_json::from_reader(reader)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // Create index_to_char lookup
    let mut index_to_char = vec!["".to_string(); vocab_map.len()];
    for (c, &idx) in &vocab_map {
        index_to_char[idx as usize] = c.clone();
    }

    println!(
        "Loaded vocabulary with {} entries in {:.2?}",
        vocab_map.len(),
        start.elapsed()
    );
    Ok((vocab_map, index_to_char))
}

// Load test input data efficiently
fn load_test_input(file_path: &str) -> io::Result<Vec<String>> {
    println!("Loading test data from {}", file_path);
    let start = Instant::now();

    let file = File::open(file_path)?;
    let metadata = fs::metadata(file_path)?;
    let file_size = metadata.len();

    println!("File size: {} bytes", file_size);

    // Process the file efficiently
    let reader = BufReader::with_capacity(8 * 1024 * 1024, file); // 8MB buffer
    let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();

    let duration = start.elapsed();
    println!("Loaded {} lines in {:.2?}", lines.len(), duration);

    Ok(lines)
}

// Embed strings for the model
fn embed_strings(inputs: &[String], vocab: &HashMap<String, i64>, pad_token: i64) -> Tensor {
    let batch_size = inputs.len() as i64;
    let mut encoded = vec![pad_token; (batch_size * CONTEXT_LENGTH) as usize];

    for (i, s) in inputs.iter().enumerate() {
        let mut indices: Vec<i64> = s
            .chars()
            .rev()
            .take(CONTEXT_LENGTH as usize)
            .map(|c| *vocab.get(&c.to_string()).unwrap_or(&pad_token))
            .collect();

        indices.reverse(); // We took chars in reverse, now flip back

        let padding_needed = CONTEXT_LENGTH as usize - indices.len();
        if padding_needed > 0 {
            let padding = vec![pad_token; padding_needed];
            indices = [padding, indices].concat();
        }

        // Copy to our flat array
        for (j, &idx) in indices.iter().enumerate().take(CONTEXT_LENGTH as usize) {
            encoded[i * CONTEXT_LENGTH as usize + j] = idx;
        }
    }

    Tensor::of_slice(&encoded)
        .reshape(&[batch_size, CONTEXT_LENGTH])
        .to(Device::Cuda)
}

// Run the model prediction
fn run_prediction(
    model: &CModule,
    inputs: &[String],
    vocab: &HashMap<String, i64>,
    index_to_char: &[String],
    pad_token: i64,
) -> Vec<String> {
    let start = Instant::now();
    println!("Preparing input tensor for {} examples", inputs.len());

    // Embed the input strings
    let input_tensor = embed_strings(inputs, vocab, pad_token);
    println!("Input tensor prepared in {:.2?}", start.elapsed());

    // Run inference
    let infer_start = Instant::now();
    println!("Running model inference...");

    let logits = model
        .forward_ts(&[input_tensor])
        .expect("Model forward pass failed");

    println!("Inference completed in {:.2?}", infer_start.elapsed());

    // Post-processing
    let post_start = Instant::now();
    let batch_size = inputs.len() as i64;

    // Set PAD_TOKEN logits to -inf
    let mut logits_masked = logits.copy();
    logits_masked.get((.., pad_token)).fill_(-f64::INFINITY);

    // Get top-3 predictions
    let (_, indices) = logits_masked.topk(3, -1, true, true);
    let indices_cpu = indices.to(Device::Cpu);

    // Convert to 2D array and then to strings
    let indices_vec: Vec<i64> = indices_cpu.to_vec1::<i64>().unwrap_or_default();

    let mut results = Vec::new();
    for batch_idx in 0..batch_size as usize {
        let start_idx = batch_idx * 3; // Each batch has 3 predictions
        let mut prediction = String::new();

        for i in 0..3 {
            let char_idx = indices_vec[start_idx + i] as usize;
            prediction.push_str(&index_to_char[char_idx]);
        }

        results.push(prediction);
    }

    println!("Post-processing completed in {:.2?}", post_start.elapsed());
    println!("Total prediction time: {:.2?}", start.elapsed());

    results
}

// Write predictions to file
fn write_predictions(predictions: &[String], file_path: &str) -> io::Result<()> {
    let start = Instant::now();

    let file = File::create(file_path)?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file); // 8MB buffer

    for pred in predictions {
        writeln!(writer, "{}", pred)?;
    }

    // Make sure all data is written
    writer.flush()?;

    let duration = start.elapsed();
    println!(
        "Wrote {} predictions in {:.2?}",
        predictions.len(),
        duration
    );

    Ok(())
}

fn main() -> io::Result<()> {
    // Parse command line arguments
    let args = parse_args();

    // Print configuration
    println!("Configuration:");
    println!("  Work directory: {}", args.work_dir);
    println!("  Test data: {}", args.test_data);
    println!("  Test output: {}", args.test_output);
    println!("  Timing enabled: {}", args.time);
    println!("  Using TorchScript: {}", args.torchscript);

    let total_start = Instant::now();

    // Step 1: Determine paths
    let model_path = if args.torchscript {
        format!("{}/character_transformer_script.pt", args.work_dir)
    } else {
        format!("{}/character_transformer.pt", args.work_dir)
    };
    let vocab_path = format!("{}/char_to_index.json", args.work_dir);

    // Step 2: Load vocabulary
    let vocab_start = Instant::now();
    let (vocab, index_to_char) = load_vocab(&vocab_path)?;
    let pad_token = *vocab.get(" ").expect("No padding token in vocabulary");

    if args.time {
        println!("Vocabulary loaded in {:.2?}", vocab_start.elapsed());
    }

    // Step 3: Load model
    let model_start = Instant::now();
    println!("Loading model from {}", model_path);

    let model = CModule::load(&model_path).map_err(|e| {
        io::Error::new(io::ErrorKind::Other, format!("Failed to load model: {}", e))
    })?;

    if args.time {
        println!("Model loaded in {:.2?}", model_start.elapsed());
    }

    // Step 4: Load test data
    let data_start = Instant::now();
    let test_input = load_test_input(&args.test_data);

    if args.time {
        println!("Data loading took {:.2?}", data_start.elapsed());
    }

    // Step 5: Run prediction
    let predict_start = Instant::now();
    let predictions = run_prediction(&model, &test_input, &vocab, &index_to_char, pad_token);

    if args.time {
        println!("Prediction took {:.2?}", predict_start.elapsed());
    }

    // Step 6: Write predictions to output file
    let write_start = Instant::now();
    write_predictions(&predictions, &args.test_output)?;

    if args.time {
        println!("Writing output took {:.2?}", write_start.elapsed());
    }

    // Report total time
    let total_duration = total_start.elapsed();
    if args.time {
        println!("\nTotal execution time: {:.2?}", total_duration);
    }

    println!("Completed successfully");
    Ok(())
}
