from perspic_utils.models.cnns import WideResNet


def show_model_comparison_info():
        """
        The WideResNet architecture was introduced here: https://arxiv.org/abs/1605.07146
        This function provides a comparison of the official WideResNet configurations
        with the implemented models in this module.

        We just use a 1/16 perspic-width-multiplier compared to the original ResNet
        to make smaller steps in parameter count.
        """
        print("="*70)
        print("WIDERESNET ARCHITECTURES COMPARISON")
        print("="*70)

        # Test cases from the paper
        configs = [
            (10, 16, "WRN-10-1 (minimal)"),
            (10, 32, "WRN-10-2"),
            (16, 16, "WRN-16-1"),
            (16, 160, "WRN-16-10"),
            (28, 160, "WRN-28-10 (main paper result)"),
            (40, 16, "WRN-40-1"),
        ]

        for depth, width_mult, name in configs:
            model = WideResNet(depth=depth, num_classes=10, widen_factor=width_mult, dropRate=0.0)
            n_params = sum(p.numel() for p in model.parameters())

            print(f"\n{name}:")
            print(f"  Depth: {depth}, perspic-width-multiplier: {width_mult}")
            print(f"  Parameters: {n_params:,}")

        print("\n" + "="*70)
        print("OFFICIAL PAPER RESULTS:")
        print("  WRN-28-10 achieves 95.19% on CIFAR-10")
        print("  WRN-40-1 achieves 95.23% on CIFAR-10")
        print("="*70)

        print("Note: perspic uses widen_factor in multiples of 16 (e.g., widen_factor=1 → 16 base channels)")
        print("      This differs from the paper's notation: WRN-10-1 requires widen_factor=16 to match parameter count")


if __name__ == "__main__":
    show_model_comparison_info()
