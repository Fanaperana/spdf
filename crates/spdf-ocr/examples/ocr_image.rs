//! Run the built-in neural OCR engine on image files and print the words.
//!
//! `cargo run --release -p spdf-ocr --features neural --example ocr_image -- page.png`

#[cfg(feature = "neural")]
fn main() {
    use spdf_ocr::{NeuralOcrEngine, OcrEngine, OcrOptions};
    let engine = NeuralOcrEngine::new();
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).expect("read image");
        let t = std::time::Instant::now();
        let words = engine
            .recognize(&bytes, &OcrOptions::default())
            .expect("ocr");
        println!(
            "== {path}: {} words in {:?} (backend {})",
            words.len(),
            t.elapsed(),
            engine.backend().unwrap_or("?")
        );
        let mut line = String::new();
        let mut last_y = f64::MIN;
        for w in &words {
            if (w.bbox[1] - last_y).abs() > 8.0 && !line.is_empty() {
                println!("{line}");
                line.clear();
            }
            last_y = w.bbox[1];
            line.push_str(&w.text);
            line.push(' ');
        }
        println!("{line}");
    }
}

#[cfg(not(feature = "neural"))]
fn main() {
    eprintln!("build with --features neural");
}
