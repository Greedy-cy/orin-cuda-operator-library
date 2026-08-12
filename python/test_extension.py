import math

import torch

from load_extension import load_operatorlib


def expect_error(fn, text: str) -> None:
    try:
        fn()
    except RuntimeError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"expected RuntimeError containing {text!r}")


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
    scores /= math.sqrt(q.size(-1))
    if causal:
        mask = torch.ones(
            q.size(-2), q.size(-2), device=q.device, dtype=torch.bool
        ).triu(1)
        scores.masked_fill_(mask, -torch.inf)
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def test_reduce_softmax(stream: torch.cuda.Stream) -> None:
    with torch.cuda.stream(stream):
        value = torch.randn(1_000_003, device="cuda", dtype=torch.float32)
        output = torch.ops.operatorlib.reduce_sum(value)
        reference = value.sum()
    stream.synchronize()
    torch.testing.assert_close(output, reference, rtol=5e-4, atol=5e-4)

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


def test_transpose_rmsnorm(stream: torch.cuda.Stream) -> None:
    for rows, cols in [(64, 96), (65, 37)]:
        with torch.cuda.stream(stream):
            value = torch.randn(rows, cols, device="cuda")
            output = torch.ops.operatorlib.transpose(value)
            reference = value.t().contiguous()
        stream.synchronize()
        torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)

    for shape in [(3, 7, 1024), (5, 1003)]:
        hidden = shape[-1]
        with torch.cuda.stream(stream):
            value = torch.randn(*shape, device="cuda")
            weight = torch.randn(hidden, device="cuda")
            output = torch.ops.operatorlib.rmsnorm(value, weight, 1e-5)
            inverse_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True)
                                      + 1e-5)
            reference = value * inverse_rms * weight
        stream.synchronize()
        torch.testing.assert_close(output, reference, rtol=2e-4, atol=2e-5)


def test_gemm(stream: torch.cuda.Stream) -> None:
    cases = [
        (torch.float32, 128, 64, 32),
        (torch.float32, 7, 37, 23),
        (torch.float16, 128, 128, 32),
        (torch.float16, 16, 128, 32),
        (torch.float16, 32, 32, 32),
        (torch.float16, 17, 19, 23),
    ]
    for dtype, m, n, k in cases:
        with torch.cuda.stream(stream):
            a = 0.1 * torch.randn(m, k, device="cuda", dtype=dtype)
            b = 0.1 * torch.randn(k, n, device="cuda", dtype=dtype)
            output = torch.ops.operatorlib.gemm(a, b)
            reference = torch.matmul(a.float(), b.float())
        stream.synchronize()
        assert output.dtype == torch.float32
        tolerance = 2e-2 if dtype == torch.float16 else 5e-4
        torch.testing.assert_close(
            output, reference, rtol=tolerance, atol=tolerance
        )


def test_attention(stream: torch.cuda.Stream) -> None:
    cases = [
        (torch.float32, 32, 64, False),
        (torch.float32, 32, 128, True),
        (torch.float32, 37, 71, True),
        (torch.float16, 32, 64, False),
        (torch.float16, 32, 128, True),
        (torch.float16, 37, 71, True),
    ]
    for dtype, sequence, head_dim, causal in cases:
        with torch.cuda.stream(stream):
            shape = (1, 2, sequence, head_dim)
            q = 0.25 * torch.randn(*shape, device="cuda", dtype=dtype)
            k = 0.25 * torch.randn(*shape, device="cuda", dtype=dtype)
            v = torch.randn(*shape, device="cuda", dtype=dtype)
            output = torch.ops.operatorlib.attention(q, k, v, causal)
            reference = attention_reference(q, k, v, causal)
        stream.synchronize()
        assert output.dtype == torch.float32
        tolerance = 2e-2 if dtype == torch.float16 else 8e-4
        torch.testing.assert_close(
            output, reference, rtol=tolerance, atol=tolerance
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
        lambda: torch.ops.operatorlib.softmax(torch.empty(3, 0, device="cuda")),
        "positive int32",
    )
    expect_error(
        lambda: torch.ops.operatorlib.transpose(torch.randn(2, 3, 4,
                                                            device="cuda")),
        "2D",
    )
    expect_error(
        lambda: torch.ops.operatorlib.rmsnorm(
            torch.randn(3, 8, device="cuda"),
            torch.randn(7, device="cuda"),
            1e-5,
        ),
        "weight length",
    )
    expect_error(
        lambda: torch.ops.operatorlib.gemm(
            torch.randn(4, 8, device="cuda"),
            torch.randn(7, 5, device="cuda"),
        ),
        "a.size(1)",
    )
    bad_shape = (1, 1, 8, 129)
    expect_error(
        lambda: torch.ops.operatorlib.attention(
            torch.randn(*bad_shape, device="cuda"),
            torch.randn(*bad_shape, device="cuda"),
            torch.randn(*bad_shape, device="cuda"),
            False,
        ),
        "at most 128",
    )


def main() -> None:
    assert torch.cuda.is_available()
    load_operatorlib(verbose=True)
    stream = torch.cuda.Stream()
    test_reduce_softmax(stream)
    test_transpose_rmsnorm(stream)
    test_gemm(stream)
    test_attention(stream)
    test_validation()
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print("operatorlib_extension=PASS")


if __name__ == "__main__":
    main()
