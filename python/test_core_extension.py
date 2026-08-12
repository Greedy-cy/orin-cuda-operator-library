import torch

from load_core_extension import load_operatorlib_core


def expect_error(fn, text: str) -> None:
    try:
        fn()
    except RuntimeError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"expected RuntimeError containing {text!r}")


def test_reduce_sum(stream: torch.cuda.Stream) -> None:
    with torch.cuda.stream(stream):
        value = torch.randn(1_000_003, device="cuda", dtype=torch.float32)
        output = torch.ops.operatorlib.reduce_sum(value)
        reference = value.sum()
    stream.synchronize()
    torch.testing.assert_close(output, reference, rtol=5e-4, atol=5e-4)


def test_softmax(stream: torch.cuda.Stream) -> None:
    for rows, cols, scale in [(19, 128, 5.0), (17, 1024, 5.0),
                              (17, 1003, 1000.0)]:
        with torch.cuda.stream(stream):
            value = scale * torch.randn(
                rows, cols, device="cuda", dtype=torch.float32
            )
            output = torch.ops.operatorlib.softmax(value)
            reference = torch.softmax(value, dim=-1)
        stream.synchronize()
        torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            output.sum(dim=-1), torch.ones(rows, device="cuda"),
            rtol=1e-5, atol=1e-5
        )


def test_validation() -> None:
    expect_error(
        lambda: torch.ops.operatorlib.reduce_sum(
            torch.ones(8, device="cuda", dtype=torch.float16)
        ),
        "float32",
    )
    non_contiguous = torch.randn(8, 16, device="cuda")[:, ::2]
    expect_error(
        lambda: torch.ops.operatorlib.softmax(non_contiguous), "contiguous"
    )
    expect_error(
        lambda: torch.ops.operatorlib.softmax(
            torch.empty(3, 0, device="cuda", dtype=torch.float32)
        ),
        "non-empty",
    )


def main() -> None:
    assert torch.cuda.is_available()
    load_operatorlib_core(verbose=True)
    stream = torch.cuda.Stream()
    test_reduce_sum(stream)
    test_softmax(stream)
    test_validation()
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print("operatorlib_core_extension=PASS")


if __name__ == "__main__":
    main()
