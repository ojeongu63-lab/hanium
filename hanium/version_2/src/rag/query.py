from .features import describe_feature


def build_query(feature_contributions: list[dict], top_n: int = 3) -> str:
    top = feature_contributions[:top_n]
    parts = [
        f"{describe_feature(c['feature'])}(z={c['z_score']:.1f})" for c in top
    ]
    return "다음 센서 값이 정상 대비 크게 벗어났습니다: " + ", ".join(parts)
