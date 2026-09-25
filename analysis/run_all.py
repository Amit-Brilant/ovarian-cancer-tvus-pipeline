"""Run every analysis in turn; an analysis whose non-public inputs are missing is reported and skipped."""
import argparse

from ovarian import seeds
from analysis import calibration_importance, clinical_baselines, readers, shap_scope, stage1_variability

ANALYSES = (readers, clinical_baselines, calibration_importance, shap_scope, stage1_variability)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    seeds.resolve(parser.parse_args(argv).seed)
    skipped = []
    for module in ANALYSES:
        name = module.__name__.split(".")[-1]
        print(f"\n== {name}")
        try:
            module.main([])
        except FileNotFoundError as error:
            print(f"skipped: {error}")
            skipped.append(name)
    print(f"\nskipped: {', '.join(skipped)}" if skipped else "\nall analyses completed")


if __name__ == "__main__":
    main()
