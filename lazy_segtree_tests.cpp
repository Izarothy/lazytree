#include "lazy_segtree.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

struct AssignTag {
    bool has;
    long long value;
};

struct AffineTag {
    long long a;
    long long b;
};

struct BitwiseAffineTag {
    std::uint32_t A;
    std::uint32_t O;
};

struct NoLazyTag {};

struct Mat2 {
    long long a00;
    long long a01;
    long long a10;
    long long a11;
};

struct SumMax {
    long long sum;
    long long mx;
};

struct BitCounts {
    std::array<int, 32> cnt;
};

static bool operator==(const Mat2& x, const Mat2& y) {
    return x.a00 == y.a00 && x.a01 == y.a01 && x.a10 == y.a10 && x.a11 == y.a11;
}

static bool operator==(const SumMax& x, const SumMax& y) {
    return x.sum == y.sum && x.mx == y.mx;
}

static bool operator==(const BitCounts& x, const BitCounts& y) {
    return x.cnt == y.cnt;
}

static Mat2 mat_mul(const Mat2& x, const Mat2& y) {
    return Mat2{
        x.a00 * y.a00 + x.a01 * y.a10,
        x.a00 * y.a01 + x.a01 * y.a11,
        x.a10 * y.a00 + x.a11 * y.a10,
        x.a10 * y.a01 + x.a11 * y.a11,
    };
}

static BitCounts bits_from_value(std::uint32_t x) {
    BitCounts s{};
    for (int b = 0; b < 32; ++b) {
        s.cnt[b] = (x >> b) & 1u;
    }
    return s;
}

static std::uint32_t value_from_point_bits(const BitCounts& s) {
    std::uint32_t x = 0;
    for (int b = 0; b < 32; ++b) {
        if (s.cnt[b] != 0) {
            x |= (1u << b);
        }
    }
    return x;
}

static std::uint64_t sum_from_bits(const BitCounts& s) {
    std::uint64_t out = 0;
    for (int b = 0; b < 32; ++b) {
        out += (std::uint64_t(1) << b) * static_cast<std::uint64_t>(s.cnt[b]);
    }
    return out;
}

template <class S, class Op, class E>
static void run_monoid_checks(const std::vector<S>& init, Op op, E e, const std::vector<std::pair<int, S>>& updates) {
    using F = NoLazyTag;
    LazySegTree<S, F> st(
        init,
        op,
        e,
        [](const F&, const S& x, int) { return x; },
        [](const F&, const F&) { return F{}; },
        []() { return F{}; });

    std::vector<S> naive = init;
    const int n = static_cast<int>(naive.size());

    auto fold = [&](int l, int r) {
        S acc = e();
        for (int i = l; i < r; ++i) {
            acc = op(acc, naive[i]);
        }
        return acc;
    };

    assert(st.all_prod() == fold(0, n));
    for (int l = 0; l <= n; ++l) {
        for (int r = l; r <= n; ++r) {
            assert(st.prod(l, r) == fold(l, r));
        }
    }

    st.apply_range(0, n, F{});
    for (int l = 0; l <= n; ++l) {
        for (int r = l; r <= n; ++r) {
            assert(st.prod(l, r) == fold(l, r));
        }
    }

    for (const auto& up : updates) {
        const int p = up.first;
        const S x = up.second;
        st.set_point(p, x);
        naive[p] = x;
        assert(st.get_point(p) == x);

        assert(st.all_prod() == fold(0, n));
        for (int l = 0; l <= n; ++l) {
            for (int r = l; r <= n; ++r) {
                assert(st.prod(l, r) == fold(l, r));
            }
        }
    }
}

static void test_sum_with_add_lazy() {
    using S = long long;
    using F = long long;

    std::vector<S> v = {1, 2, 3, 4, 5, 6, 7, 8};
    LazySegTree<S, F> st(
        v,
        [](const S& a, const S& b) { return a + b; },
        []() { return 0LL; },
        [](const F& f, const S& x, int len) { return x + f * len; },
        [](const F& f, const F& g) { return f + g; },
        []() { return 0LL; });

    assert(st.all_prod() == 36);
    assert(st.prod(0, 8) == 36);
    assert(st.prod(2, 6) == 18);

    st.apply_range(2, 6, 5);
    assert(st.prod(0, 8) == 56);
    assert(st.get_point(2) == 8);
    assert(st.get_point(5) == 11);

    st.set_point(3, 100);
    assert(st.get_point(3) == 100);
    assert(st.prod(3, 4) == 100);
    assert(st.prod(0, 8) == 147);
}

static void test_max_with_assign_lazy() {
    using S = long long;
    using F = AssignTag;

    constexpr S NEG_INF = std::numeric_limits<S>::lowest() / 4;
    std::vector<S> v = {5, 1, 9, 2, 7, 3};

    LazySegTree<S, F> st(
        v,
        [](const S& a, const S& b) { return std::max(a, b); },
        [NEG_INF]() { return NEG_INF; },
        [](const F& f, const S& x, int) { return f.has ? f.value : x; },
        [](const F& f, const F& g) { return f.has ? f : g; },
        []() { return AssignTag{false, 0}; });

    assert(st.prod(0, 6) == 9);
    st.apply_range(1, 5, AssignTag{true, 4});
    assert(st.prod(0, 6) == 5);
    assert(st.prod(1, 5) == 4);

    st.apply_range(2, 6, AssignTag{true, 10});
    assert(st.prod(0, 6) == 10);
    assert(st.get_point(2) == 10);
    assert(st.get_point(5) == 10);
}

static void stress_test_sum_add_lazy_against_naive() {
    using S = long long;
    using F = long long;

    constexpr int N = 128;
    constexpr int STEPS = 6000;

    std::mt19937 rng(123456789u);
    std::uniform_int_distribution<int> dist_val(-50, 50);
    std::uniform_int_distribution<int> dist_op(0, 3);
    std::uniform_int_distribution<int> dist_idx(0, N - 1);

    std::vector<S> naive(N, 0);
    LazySegTree<S, F> st(
        naive,
        [](const S& a, const S& b) { return a + b; },
        []() { return 0LL; },
        [](const F& f, const S& x, int len) { return x + f * len; },
        [](const F& f, const F& g) { return f + g; },
        []() { return 0LL; });

    for (int step = 0; step < STEPS; ++step) {
        const int op = dist_op(rng);
        int l = dist_idx(rng);
        int r = dist_idx(rng);
        if (l > r) {
            std::swap(l, r);
        }
        ++r;

        if (op == 0) {
            const long long delta = dist_val(rng);
            st.apply_range(l, r, delta);
            for (int i = l; i < r; ++i) {
                naive[i] += delta;
            }
        } else if (op == 1) {
            S expected = 0;
            for (int i = l; i < r; ++i) {
                expected += naive[i];
            }
            const S got = st.prod(l, r);
            assert(got == expected);
        } else if (op == 2) {
            const int p = dist_idx(rng);
            const S x = dist_val(rng);
            st.set_point(p, x);
            naive[p] = x;
        } else {
            const int p = dist_idx(rng);
            assert(st.get_point(p) == naive[p]);
        }
    }

    S expected_all = 0;
    for (const S x : naive) {
        expected_all += x;
    }
    assert(st.all_prod() == expected_all);
}

static void test_sum_with_affine_lazy() {
    using S = long long;
    using F = AffineTag;

    std::vector<S> v = {1, 2, 3, 4, 5};
    LazySegTree<S, F> st(
        v,
        [](const S& x, const S& y) { return x + y; },
        []() { return 0LL; },
        [](const F& f, const S& sum, int len) { return f.a * sum + f.b * len; },
        [](const F& f, const F& g) { return F{f.a * g.a, f.a * g.b + f.b}; },
        []() { return F{1, 0}; });

    assert(st.prod(0, 5) == 15);

    st.apply_range(1, 4, F{2, 1});  // [1,2,3,4,5] -> [1,5,7,9,5]
    assert(st.prod(0, 5) == 27);
    assert(st.prod(1, 4) == 21);

    st.apply_range(2, 5, F{-1, 3}); // [1,5,7,9,5] -> [1,5,-4,-6,-2]
    assert(st.prod(0, 5) == -6);
    assert(st.get_point(2) == -4);
    assert(st.get_point(4) == -2);

    st.set_point(3, 10);            // [1,5,-4,10,-2]
    assert(st.prod(0, 5) == 10);
    assert(st.prod(2, 5) == 4);
}

static void stress_test_sum_affine_lazy_against_naive() {
    using S = long long;
    using F = AffineTag;

    constexpr int N = 96;
    constexpr int STEPS = 7000;

    std::mt19937 rng(987654321u);
    std::uniform_int_distribution<int> dist_op(0, 3);
    std::uniform_int_distribution<int> dist_idx(0, N - 1);
    std::uniform_int_distribution<int> dist_set(-100, 100);
    std::uniform_int_distribution<int> dist_a(-1, 1);
    std::uniform_int_distribution<int> dist_b(-10, 10);

    std::vector<S> naive(N, 0);
    LazySegTree<S, F> st(
        naive,
        [](const S& x, const S& y) { return x + y; },
        []() { return 0LL; },
        [](const F& f, const S& sum, int len) { return f.a * sum + f.b * len; },
        [](const F& f, const F& g) { return F{f.a * g.a, f.a * g.b + f.b}; },
        []() { return F{1, 0}; });

    for (int step = 0; step < STEPS; ++step) {
        int l = dist_idx(rng);
        int r = dist_idx(rng);
        if (l > r) {
            std::swap(l, r);
        }
        ++r;

        const int op = dist_op(rng);
        if (op == 0) {
            const F f{dist_a(rng), dist_b(rng)};
            st.apply_range(l, r, f);
            for (int i = l; i < r; ++i) {
                naive[i] = f.a * naive[i] + f.b;
            }
        } else if (op == 1) {
            S expected = 0;
            for (int i = l; i < r; ++i) {
                expected += naive[i];
            }
            assert(st.prod(l, r) == expected);
        } else if (op == 2) {
            const int p = dist_idx(rng);
            const S x = dist_set(rng);
            st.set_point(p, x);
            naive[p] = x;
        } else {
            const int p = dist_idx(rng);
            assert(st.get_point(p) == naive[p]);
        }
    }

    S expected_all = 0;
    for (const S x : naive) {
        expected_all += x;
    }
    assert(st.all_prod() == expected_all);
}

static void test_sum_with_bitwise_affine_lazy() {
    using S = BitCounts;
    using F = BitwiseAffineTag;

    std::vector<std::uint32_t> base = {
        0b1011u,
        0b0101u,
        0b1110u,
        0b0011u,
        0b1000u,
    };
    std::vector<S> init;
    init.reserve(base.size());
    for (std::uint32_t x : base) {
        init.push_back(bits_from_value(x));
    }

    LazySegTree<S, F> st(
        init,
        [](const S& a, const S& b) {
            S out{};
            for (int i = 0; i < 32; ++i) {
                out.cnt[i] = a.cnt[i] + b.cnt[i];
            }
            return out;
        },
        []() {
            return S{};
        },
        [](const F& f, const S& x, int len) {
            S out{};
            for (int i = 0; i < 32; ++i) {
                const bool o = ((f.O >> i) & 1u) != 0;
                const bool a = ((f.A >> i) & 1u) != 0;
                out.cnt[i] = o ? len : (a ? x.cnt[i] : 0);
            }
            return out;
        },
        [](const F& f, const F& g) {
            return F{f.A & g.A, (g.O & f.A) | f.O};
        },
        []() { return F{~0u, 0u}; });

    std::vector<std::uint32_t> naive = base;
    auto range_sum = [&](int l, int r) {
        std::uint64_t s = 0;
        for (int i = l; i < r; ++i) {
            s += naive[i];
        }
        return s;
    };

    assert(sum_from_bits(st.all_prod()) == range_sum(0, static_cast<int>(naive.size())));

    st.apply_range(1, 5, F{0b1110u, 0b0001u});
    for (int i = 1; i < 5; ++i) {
        naive[i] = (naive[i] & 0b1110u) | 0b0001u;
    }
    assert(sum_from_bits(st.prod(0, 5)) == range_sum(0, 5));
    assert(sum_from_bits(st.prod(1, 4)) == range_sum(1, 4));

    st.apply_range(0, 3, F{0b1011u, 0b0100u});
    for (int i = 0; i < 3; ++i) {
        naive[i] = (naive[i] & 0b1011u) | 0b0100u;
    }
    assert(sum_from_bits(st.prod(0, 5)) == range_sum(0, 5));
    assert(sum_from_bits(st.prod(0, 3)) == range_sum(0, 3));

    st.set_point(2, bits_from_value(0b0010u));
    naive[2] = 0b0010u;
    assert(value_from_point_bits(st.get_point(2)) == naive[2]);
    assert(sum_from_bits(st.all_prod()) == range_sum(0, 5));
}

static void stress_test_sum_bitwise_affine_lazy_against_naive() {
    using S = BitCounts;
    using F = BitwiseAffineTag;

    constexpr int N = 80;
    constexpr int STEPS = 7000;

    std::mt19937 rng(246813579u);
    std::uniform_int_distribution<int> dist_op(0, 4);
    std::uniform_int_distribution<int> dist_idx(0, N - 1);
    std::uniform_int_distribution<std::uint32_t> dist_u32(0u, 0xFFFFu);

    std::vector<std::uint32_t> naive(N);
    for (int i = 0; i < N; ++i) {
        naive[i] = dist_u32(rng);
    }

    std::vector<S> init;
    init.reserve(N);
    for (std::uint32_t x : naive) {
        init.push_back(bits_from_value(x));
    }

    LazySegTree<S, F> st(
        init,
        [](const S& a, const S& b) {
            S out{};
            for (int i = 0; i < 32; ++i) {
                out.cnt[i] = a.cnt[i] + b.cnt[i];
            }
            return out;
        },
        []() {
            return S{};
        },
        [](const F& f, const S& x, int len) {
            S out{};
            for (int i = 0; i < 32; ++i) {
                const bool o = ((f.O >> i) & 1u) != 0;
                const bool a = ((f.A >> i) & 1u) != 0;
                out.cnt[i] = o ? len : (a ? x.cnt[i] : 0);
            }
            return out;
        },
        [](const F& f, const F& g) {
            return F{f.A & g.A, (g.O & f.A) | f.O};
        },
        []() { return F{~0u, 0u}; });

    auto expected_bits = [&](int l, int r) {
        S out{};
        for (int i = l; i < r; ++i) {
            for (int b = 0; b < 32; ++b) {
                out.cnt[b] += (naive[i] >> b) & 1u;
            }
        }
        return out;
    };

    for (int step = 0; step < STEPS; ++step) {
        int l = dist_idx(rng);
        int r = dist_idx(rng);
        if (l > r) {
            std::swap(l, r);
        }
        ++r;

        const int op = dist_op(rng);
        if (op == 0) {
            const F f{dist_u32(rng), dist_u32(rng)};
            st.apply_range(l, r, f);
            for (int i = l; i < r; ++i) {
                naive[i] = (naive[i] & f.A) | f.O;
            }
        } else if (op == 1) {
            assert(st.prod(l, r) == expected_bits(l, r));
        } else if (op == 2) {
            const int p = dist_idx(rng);
            const std::uint32_t x = dist_u32(rng);
            st.set_point(p, bits_from_value(x));
            naive[p] = x;
        } else if (op == 3) {
            const int p = dist_idx(rng);
            assert(value_from_point_bits(st.get_point(p)) == naive[p]);
        } else {
            assert(st.all_prod() == expected_bits(0, N));
        }
    }
}

static void test_monoid_sum_tree() {
    using S = long long;
    run_monoid_checks<S>(
        {3, -1, 4, 1, 5, -9, 2, 6},
        [](const S& a, const S& b) { return a + b; },
        []() { return 0LL; },
        {{0, 10}, {5, 7}, {7, -3}});
}

static void test_monoid_min_tree() {
    using S = long long;
    run_monoid_checks<S>(
        {7, 2, 9, 4, 8, 1, 6},
        [](const S& a, const S& b) { return std::min(a, b); },
        []() { return std::numeric_limits<S>::max(); },
        {{3, -5}, {1, 11}, {5, 0}});
}

static void test_monoid_max_tree() {
    using S = long long;
    run_monoid_checks<S>(
        {7, 2, 9, 4, 8, 1, 6},
        [](const S& a, const S& b) { return std::max(a, b); },
        []() { return std::numeric_limits<S>::lowest(); },
        {{3, -5}, {1, 11}, {5, 0}});
}

static void test_monoid_gcd_tree() {
    using S = long long;
    run_monoid_checks<S>(
        {12, 18, 30, 42, 54, 66},
        [](const S& a, const S& b) { return std::gcd(a, b); },
        []() { return 0LL; },
        {{0, 24}, {2, 45}, {5, 81}});
}

static void test_monoid_xor_tree() {
    using S = std::uint32_t;
    run_monoid_checks<S>(
        {0x1u, 0x3u, 0x7u, 0xFu, 0x10u, 0x55u},
        [](const S& a, const S& b) { return a ^ b; },
        []() { return 0u; },
        {{1, 0xABu}, {3, 0x33u}, {4, 0x0u}});
}

static void test_monoid_or_tree() {
    using S = std::uint32_t;
    run_monoid_checks<S>(
        {0x1u, 0x2u, 0x4u, 0x8u, 0x10u},
        [](const S& a, const S& b) { return a | b; },
        []() { return 0u; },
        {{0, 0x40u}, {2, 0x80u}, {4, 0x0u}});
}

static void test_monoid_and_tree() {
    using S = std::uint32_t;
    run_monoid_checks<S>(
        {0xFFu, 0xF0u, 0xCCu, 0xAAu, 0x0Fu},
        [](const S& a, const S& b) { return a & b; },
        []() { return ~0u; },
        {{0, 0x3Fu}, {1, 0x55u}, {4, 0xF3u}});
}

static void test_monoid_lcm_tree() {
    using S = std::uint64_t;
    run_monoid_checks<S>(
        {2ull, 3ull, 4ull, 5ull, 6ull},
        [](const S& a, const S& b) { return std::lcm(a, b); },
        []() { return 1ull; },
        {{0, 7ull}, {2, 9ull}, {4, 10ull}});
}

static void test_monoid_matrix_product_tree() {
    using S = Mat2;
    const std::vector<S> init = {
        {1, 2, 3, 4},
        {0, 1, 1, 0},
        {2, 1, 0, 1},
        {1, 0, 2, 1},
    };
    run_monoid_checks<S>(
        init,
        [](const S& a, const S& b) { return mat_mul(a, b); },
        []() { return Mat2{1, 0, 0, 1}; },
        {{1, Mat2{2, 0, 1, 2}}, {3, Mat2{1, 1, 0, 1}}});
}

static void test_monoid_pair_tree() {
    using S = SumMax;
    run_monoid_checks<S>(
        {
            {3, 3},
            {1, 1},
            {4, 4},
            {1, 1},
            {5, 5},
            {9, 9},
        },
        [](const S& a, const S& b) { return S{a.sum + b.sum, std::max(a.mx, b.mx)}; },
        []() { return S{0, std::numeric_limits<long long>::lowest()}; },
        {{1, S{10, 10}}, {4, S{-2, -2}}, {5, S{7, 7}}});
}

int main() {
    test_sum_with_add_lazy();
    test_sum_with_affine_lazy();
    test_sum_with_bitwise_affine_lazy();
    test_max_with_assign_lazy();
    stress_test_sum_add_lazy_against_naive();
    stress_test_sum_affine_lazy_against_naive();
    stress_test_sum_bitwise_affine_lazy_against_naive();

    test_monoid_sum_tree();
    test_monoid_min_tree();
    test_monoid_max_tree();
    test_monoid_gcd_tree();
    test_monoid_xor_tree();
    test_monoid_or_tree();
    test_monoid_and_tree();
    test_monoid_lcm_tree();
    test_monoid_matrix_product_tree();
    test_monoid_pair_tree();

    std::cout << "All LazySegTree tests passed.\n";
    return 0;
}
