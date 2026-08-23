import argparse
import gc

from models.model_configs import MODEL_CONFIGS_RECOMMENDED, instantiate_model


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def count_parameters(model) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def print_table(rows: list[tuple[str, int, int]]) -> None:
    headers = ("model", "total", "trainable", "total_readable", "trainable_readable")
    table = [
        (name, str(total), str(trainable), format_count(total), format_count(trainable))
        for name, total, trainable in rows
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table))
        for i in range(len(headers))
    ]

    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count parameters for models defined in models/model_configs.py."
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Count parameters after wrapping the model with EMA.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS_RECOMMENDED.keys()),
        help="Only count one model architecture.",
    )
    args = parser.parse_args()

    names = [args.model] if args.model else list(MODEL_CONFIGS_RECOMMENDED.keys())
    rows = []
    for name in names:
        model = instantiate_model(name, use_ema=args.ema)
        total, trainable = count_parameters(model)
        rows.append((name, total, trainable))
        del model
        gc.collect()

    print_table(rows)


if __name__ == "__main__":
    main()
