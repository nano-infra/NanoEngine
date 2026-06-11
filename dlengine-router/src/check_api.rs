use tokenizers::Tokenizer;

fn main() {
    let t = Tokenizer::from_file("tokenizer.json").unwrap();
    // Check if get_chat_template exists
    let template = t.get_chat_template();
    println!("{:?}", template);
}
