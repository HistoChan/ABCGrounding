def generate_theory() -> None:
    from data_parsing import parse_data_files
    from theory_construction import python_coupling

    # 1. Load all knowledge and label them
    parse_data_files("../webnlg", "webnlg")
    parse_data_files("../e2e", "dart")
    parse_data_files("../manual", "dart")

    # 2. construct theories
    categories = [
        "Airport",
        "Artist",
        "Astronaut",
        "Athlete",
        "Building",
        "CelestialBody",
        "City",
        "ComicsCharacter",
        "Food",
        "MeanOfTransportation",
        "Monument",
        "Politician",
        "SportsTeam",
        "University",
        "WrittenWork",
    ]
    for category in categories:
        python_coupling("./preprocessed_webnlg.pickle", category)
    python_coupling("./preprocessed_e2e.pickle", "e2e")
    categories = ["WikiTableQuestions_lily", "WikiSQL_lily", "WikiTableQuestions_mturk"]
    for category in categories:
        python_coupling("./preprocessed_manual.pickle", category)


def generate_experiment(home_dir: str) -> None:
    with open("experiment.txt", "w", encoding="utf-8") as fw:
        model_types = [
            "dolly-v2-3b",
            "t5-large-ssm-nq",
            "open_llama_3b",
        ]
        for model_type in model_types:
            fw.write(f"python main.py {home_dir} {model_type} 3\n")