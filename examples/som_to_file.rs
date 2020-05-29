use kohonen::calc::neighborhood::Neighborhood;
use kohonen::map::som::DecayParam;
use kohonen::proc::{InputLayer, ProcessorBuilder};

fn main() {
    let layers = vec![
        InputLayer::cont_simple(&[
            "child_mort_2010",
            "birth_p_1000",
            "log_GNI",
            "LifeExpectancy",
            "PopGrowth",
            "PopUrbanized",
            "PopGrowthUrb",
            "AdultLiteracy",
            "PrimSchool",
            "Income_low_40",
            "Income_high_20",
        ]),
        InputLayer::cat_simple("continent"),
    ];

    let proc = ProcessorBuilder::new(
        &layers,
        &["Country".to_string(), "code".to_string()],
        &Some("Country".to_string()),
        &Some(12),
        &None,
    )
    .with_delimiter(b';')
    .with_no_data("-")
    .build_from_file("example_data/countries.csv")
    .unwrap();

    let _som = proc.create_som(
        16,
        20,
        1000,
        Neighborhood::Gauss,
        DecayParam::lin(0.2, 0.01),
        DecayParam::lin(8.0, 0.5),
        DecayParam::exp(0.2, 0.001),
    );
    /*
    let serialized = serde_json::to_string(&(som, proc.denorm())).unwrap();
    let mut file = File::create("test.json").unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
    */
}
