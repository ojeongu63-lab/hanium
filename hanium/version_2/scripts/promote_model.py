import argparse

from lstm_ae.tracking import promote_to_champion


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote an MLflow model version to champion")
    parser.add_argument("version", help="Registered model version number to promote")
    args = parser.parse_args()
    promote_to_champion(args.version)
    print(f"promoted version {args.version} to champion")


if __name__ == "__main__":
    main()
