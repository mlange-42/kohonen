use csv::StringRecord;

fn main() {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path("example_data/iris.csv")
        .unwrap();
    let header: StringRecord = reader.headers().unwrap().clone();
    let _header: Vec<_> = header.iter().collect();

    /*
    println!("{:?}", header);
    for record in reader.records() {
        println!("{:?}", record);
    }
    */
}
