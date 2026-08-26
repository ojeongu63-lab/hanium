from rag.features import describe_feature


def test_describe_feature_known():
    assert describe_feature("S_OutputCurrent") == "스핀들 출력 전류"


def test_describe_feature_unknown_falls_back_to_code():
    assert describe_feature("UNKNOWN_SENSOR") == "UNKNOWN_SENSOR"
