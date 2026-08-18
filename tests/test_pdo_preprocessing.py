import pandas as pd

from preprocess_pdo_caf_mendeley_full import filter_pdo_conditions


def test_filter_pdo_conditions_matches_requested_definition() -> None:
    frame = pd.DataFrame(
        {
            "Patient": [21, 21, 75, 99, 23, 27, 21],
            "Treatment": ["DMSO", "DMSO", "5-FU", "SN-38", "O", "H2O", "DMSO"],
            "Concentration": [0, 0, 1, 2, 3, 0, 2],
            "Culture": ["PDO", "CAF", "PDOF", "PDOF", "PDO", "PDOF", "PDO"],
            "Cell_type": ["PDOs", "PDOs", "PDOs", "Fibs", "PDOs", "PDOs", "PDOs"],
            "Replicate": ["A", "A", "B", "C", "A", "B", "B"],
        }
    )

    result = filter_pdo_conditions(frame)

    assert result.index.tolist() == [2, 6]
    assert result["Condition"].tolist() == [
        "75*5-FU*1*PDOF*PDOs_B",
        "21*DMSO*2*PDO*PDOs_B",
    ]
