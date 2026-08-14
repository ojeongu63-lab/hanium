from preprocessing.data_io import load_csv


def test_load_csv_preserves_id_as_string(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("_id,value\n0001,1.5\n0002,2.5\n")

    df = load_csv(str(csv_path))

    assert df["_id"].tolist() == ["0001", "0002"]
    assert df["value"].tolist() == [1.5, 2.5]
