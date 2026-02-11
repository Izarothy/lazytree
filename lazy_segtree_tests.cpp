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

struct AssignAffineTag {
    bool has_assign;
    long long assign;
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

struct Mat3 {
    std::array<std::array<long long, 3>, 3> a;
};

struct SumMax {
    long long sum;
    long long mx;
};

struct MinMax {
    long long mn;
    long long mx;
};

struct BitCounts {
    std::array<int, 32> cnt;
};

struct MatConjAssignTag {
    bool has_assign;
    Mat2 assign;
    Mat2 l;
    Mat2 r;
};

struct PolyNode {
    long long sum;
    long long cnt;
    long long si;
    long long si2;
};

struct PolyTag {
    long long a;
    long long c0;
    long long c1;
    long long c2;
};

static bool operator==(const Mat2& x, const Mat2& y) {
    return x.a00 == y.a00 && x.a01 == y.a01 && x.a10 == y.a10 && x.a11 == y.a11;
}

static bool operator==(const Mat3& x, const Mat3& y) {
    return x.a == y.a;
}

static bool operator==(const SumMax& x, const SumMax& y) {
    return x.sum == y.sum && x.mx == y.mx;
}

static bool operator==(const MinMax& x, const MinMax& y) {
    return x.mn == y.mn && x.mx == y.mx;
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

static Mat3 mat3_mul(const Mat3& x, const Mat3& y) {
    Mat3 out{{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}}};
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j) {
                out.a[i][j] += x.a[i][k] * y.a[k][j];
            }
        }
    }
    return out;
}

static PolyNode poly_point(long long x, int i) {
    const long long ii = i;
    return PolyNode{x, 1, ii, ii * ii};
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

static MinMax minmax_point(long long x) {
    return MinMax{x, x};
}

static std::pair<long long, long long> assign_affine_linear(const AssignAffineTag& f) {
    if (f.has_assign) {
        return {0, f.a * f.assign + f.b};
    }
    return {f.a, f.b};
}

static AssignAffineTag assign_affine_from_linear(long long alpha, long long beta) {
    if (alpha == 0) {
        return AssignAffineTag{true, 0, 0, beta};
    }
    return AssignAffineTag{false, 0, alpha, beta};
}

static constexpr long long MOD = 1000000007LL;

template <class S>
static S map_no_lazy(NoLazyTag, S x, int) {
    return x;
}

static NoLazyTag compose_no_lazy(NoLazyTag, NoLazyTag) {
    return NoLazyTag{};
}

static NoLazyTag id_no_lazy() {
    return NoLazyTag{};
}

static long long op_sum_ll(long long a, long long b) {
    return a + b;
}

static long long e_sum_ll() {
    return 0LL;
}

static long long map_add_ll(long long f, long long x, int len) {
    return x + f * len;
}

static long long compose_add_ll(long long f, long long g) {
    return f + g;
}

static long long id_add_ll() {
    return 0LL;
}

static long long op_max_ll(long long a, long long b) {
    return std::max(a, b);
}

static long long e_max_ll() {
    return std::numeric_limits<long long>::lowest() / 4;
}

static long long map_assign_tag(AssignTag f, long long x, int) {
    return f.has ? f.value : x;
}

static AssignTag compose_assign_tag(AssignTag f, AssignTag g) {
    return f.has ? f : g;
}

static AssignTag id_assign_tag() {
    return AssignTag{false, 0};
}

static long long map_affine_sum(AffineTag f, long long sum, int len) {
    return f.a * sum + f.b * len;
}

static AffineTag compose_affine(AffineTag f, AffineTag g) {
    return AffineTag{f.a * g.a, f.a * g.b + f.b};
}

static AffineTag id_affine() {
    return AffineTag{1, 0};
}

static long long map_assign_affine_sum(AssignAffineTag f, long long sum, int len) {
    const auto p = assign_affine_linear(f);
    return p.first * sum + p.second * len;
}

static AssignAffineTag compose_assign_affine(AssignAffineTag f, AssignAffineTag g) {
    const auto nf = assign_affine_linear(f);
    const auto ng = assign_affine_linear(g);
    return assign_affine_from_linear(nf.first * ng.first, nf.first * ng.second + nf.second);
}

static AssignAffineTag id_assign_affine() {
    return AssignAffineTag{false, 0, 1, 0};
}

static MinMax op_minmax(MinMax a, MinMax b) {
    return MinMax{std::min(a.mn, b.mn), std::max(a.mx, b.mx)};
}

static MinMax e_minmax() {
    constexpr long long inf = std::numeric_limits<long long>::max() / 4;
    return MinMax{inf, -inf};
}

static MinMax map_affine_minmax(AffineTag f, MinMax x, int) {
    if (f.a >= 0) {
        return MinMax{f.a * x.mn + f.b, f.a * x.mx + f.b};
    }
    return MinMax{f.a * x.mx + f.b, f.a * x.mn + f.b};
}

static BitCounts op_bitcounts(BitCounts a, BitCounts b) {
    BitCounts out{};
    for (int i = 0; i < 32; ++i) {
        out.cnt[i] = a.cnt[i] + b.cnt[i];
    }
    return out;
}

static BitCounts e_bitcounts() {
    return BitCounts{};
}

static BitCounts map_bitwise_affine(BitwiseAffineTag f, BitCounts x, int len) {
    BitCounts out{};
    for (int i = 0; i < 32; ++i) {
        const bool o = ((f.O >> i) & 1u) != 0;
        const bool a = ((f.A >> i) & 1u) != 0;
        out.cnt[i] = o ? len : (a ? x.cnt[i] : 0);
    }
    return out;
}

static BitwiseAffineTag compose_bitwise_affine(BitwiseAffineTag f, BitwiseAffineTag g) {
    return BitwiseAffineTag{f.A & g.A, (g.O & f.A) | f.O};
}

static BitwiseAffineTag id_bitwise_affine() {
    return BitwiseAffineTag{~0u, 0u};
}

static std::uint32_t op_xor_u32(std::uint32_t a, std::uint32_t b) {
    return a ^ b;
}

static std::uint32_t e_xor_u32() {
    return 0u;
}

static std::uint32_t map_xor_u32(std::uint32_t f, std::uint32_t x, int len) {
    return (len & 1) ? (x ^ f) : x;
}

static std::uint32_t compose_xor_u32(std::uint32_t f, std::uint32_t g) {
    return f ^ g;
}

static std::uint32_t id_xor_u32() {
    return 0u;
}

static long long mod_norm(long long x) {
    x %= MOD;
    if (x < 0) {
        x += MOD;
    }
    return x;
}

static Mat2 mat_mul_mod(Mat2 a, Mat2 b) {
    return Mat2{
        mod_norm(a.a00 * b.a00 + a.a01 * b.a10),
        mod_norm(a.a00 * b.a01 + a.a01 * b.a11),
        mod_norm(a.a10 * b.a00 + a.a11 * b.a10),
        mod_norm(a.a10 * b.a01 + a.a11 * b.a11),
    };
}

static Mat2 mat_id_mod() {
    return Mat2{1, 0, 0, 1};
}

static Mat2 mat_pow_mod(Mat2 base, int exp) {
    Mat2 out = mat_id_mod();
    while (exp > 0) {
        if (exp & 1) {
            out = mat_mul_mod(out, base);
        }
        base = mat_mul_mod(base, base);
        exp >>= 1;
    }
    return out;
}

static Mat2 mat_apply_point_mod(MatConjAssignTag f, Mat2 x) {
    Mat2 v = f.has_assign ? f.assign : x;
    v = mat_mul_mod(f.l, v);
    v = mat_mul_mod(v, f.r);
    return v;
}

static Mat2 op_mat2_mod(Mat2 a, Mat2 b) {
    return mat_mul_mod(a, b);
}

static Mat2 e_mat2_mod() {
    return mat_id_mod();
}

static Mat2 map_mat_conj_assign_mod(MatConjAssignTag f, Mat2 p, int len) {
    if (f.has_assign) {
        const Mat2 k = mat_apply_point_mod(f, Mat2{0, 0, 0, 0});
        return mat_pow_mod(k, len);
    }
    return mat_mul_mod(mat_mul_mod(f.l, p), f.r);
}

static MatConjAssignTag compose_mat_conj_assign_mod(MatConjAssignTag f, MatConjAssignTag g) {
    if (f.has_assign) {
        return f;
    }
    if (g.has_assign) {
        return MatConjAssignTag{true, g.assign, mat_mul_mod(f.l, g.l), mat_mul_mod(g.r, f.r)};
    }
    return MatConjAssignTag{false, Mat2{0, 0, 0, 0}, mat_mul_mod(f.l, g.l), mat_mul_mod(g.r, f.r)};
}

static MatConjAssignTag id_mat_conj_assign_mod() {
    return MatConjAssignTag{false, Mat2{0, 0, 0, 0}, Mat2{1, 0, 0, 1}, Mat2{1, 0, 0, 1}};
}

static PolyNode op_poly_node(PolyNode a, PolyNode b) {
    return PolyNode{a.sum + b.sum, a.cnt + b.cnt, a.si + b.si, a.si2 + b.si2};
}

static PolyNode e_poly_node() {
    return PolyNode{0, 0, 0, 0};
}

static PolyNode map_poly(PolyTag f, PolyNode x, int) {
    return PolyNode{
        f.a * x.sum + f.c0 * x.cnt + f.c1 * x.si + f.c2 * x.si2,
        x.cnt,
        x.si,
        x.si2,
    };
}

static PolyTag compose_poly(PolyTag f, PolyTag g) {
    return PolyTag{
        f.a * g.a,
        f.a * g.c0 + f.c0,
        f.a * g.c1 + f.c1,
        f.a * g.c2 + f.c2,
    };
}

static PolyTag id_poly() {
    return PolyTag{1, 0, 0, 0};
}

static long long op_min_ll(long long a, long long b) {
    return std::min(a, b);
}

static long long e_min_ll() {
    return std::numeric_limits<long long>::max();
}

static long long op_gcd_ll(long long a, long long b) {
    return std::gcd(a, b);
}

static std::uint32_t op_or_u32(std::uint32_t a, std::uint32_t b) {
    return a | b;
}

static std::uint32_t e_or_u32() {
    return 0u;
}

static std::uint32_t op_and_u32(std::uint32_t a, std::uint32_t b) {
    return a & b;
}

static std::uint32_t e_and_u32() {
    return ~0u;
}

static std::uint64_t op_lcm_u64(std::uint64_t a, std::uint64_t b) {
    return std::lcm(a, b);
}

static std::uint64_t e_lcm_u64() {
    return 1ull;
}

static Mat2 op_mat2_plain(Mat2 a, Mat2 b) {
    return mat_mul(a, b);
}

static Mat2 e_mat2_plain() {
    return Mat2{1, 0, 0, 1};
}

static Mat3 op_mat3_plain(Mat3 a, Mat3 b) {
    return mat3_mul(a, b);
}

static Mat3 e_mat3_plain() {
    return Mat3{{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}};
}

static SumMax op_summax(SumMax a, SumMax b) {
    return SumMax{a.sum + b.sum, std::max(a.mx, b.mx)};
}

static SumMax e_summax() {
    return SumMax{0, std::numeric_limits<long long>::lowest()};
}

using SumAddTree = LazySegTree<long long, op_sum_ll, e_sum_ll, long long, map_add_ll, compose_add_ll, id_add_ll>;
using MaxAssignTree = LazySegTree<long long, op_max_ll, e_max_ll, AssignTag, map_assign_tag, compose_assign_tag, id_assign_tag>;
using SumAffineTree = LazySegTree<long long, op_sum_ll, e_sum_ll, AffineTag, map_affine_sum, compose_affine, id_affine>;
using SumAssignAffineTree = LazySegTree<long long, op_sum_ll, e_sum_ll, AssignAffineTag, map_assign_affine_sum, compose_assign_affine, id_assign_affine>;
using SumAssignAffineTreeStatic = LazySegTree<long long, op_sum_ll, e_sum_ll, AssignAffineTag, map_assign_affine_sum, compose_assign_affine, id_assign_affine, 256>;
using MinMaxAffineTree = LazySegTree<MinMax, op_minmax, e_minmax, AffineTag, map_affine_minmax, compose_affine, id_affine>;
using BitwiseAffineTree = LazySegTree<BitCounts, op_bitcounts, e_bitcounts, BitwiseAffineTag, map_bitwise_affine, compose_bitwise_affine, id_bitwise_affine>;
using XorLazyTree = LazySegTree<std::uint32_t, op_xor_u32, e_xor_u32, std::uint32_t, map_xor_u32, compose_xor_u32, id_xor_u32>;
using MatrixAssignAffineTree = LazySegTree<Mat2, op_mat2_mod, e_mat2_mod, MatConjAssignTag, map_mat_conj_assign_mod, compose_mat_conj_assign_mod, id_mat_conj_assign_mod>;
using PolyTree = LazySegTree<PolyNode, op_poly_node, e_poly_node, PolyTag, map_poly, compose_poly, id_poly>;

template <class S, S (*OpFn)(S, S), S (*EFn)()>
static void run_monoid_checks(const std::vector<S>& init, const std::vector<std::pair<int, S>>& updates) {
    using Tree = LazySegTree<S, OpFn, EFn, NoLazyTag, map_no_lazy<S>, compose_no_lazy, id_no_lazy>;
    Tree st(init);

    std::vector<S> naive = init;
    const int n = static_cast<int>(naive.size());

    auto fold = [&](int l, int r) {
        S acc = EFn();
        for (int i = l; i < r; ++i) {
            acc = OpFn(acc, naive[i]);
        }
        return acc;
    };

    assert(st.all_prod() == fold(0, n));
    for (int l = 0; l <= n; ++l) {
        for (int r = l; r <= n; ++r) {
            assert(st.prod(l, r) == fold(l, r));
        }
    }

    st.apply_range(0, n, NoLazyTag{});
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

    std::vector<S> v = {1, 2, 3, 4, 5, 6, 7, 8};
    SumAddTree st(v);

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
    std::vector<S> v = {5, 1, 9, 2, 7, 3};
    MaxAssignTree st(v);

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

    constexpr int N = 128;
    constexpr int STEPS = 6000;

    std::mt19937 rng(123456789u);
    std::uniform_int_distribution<int> dist_val(-50, 50);
    std::uniform_int_distribution<int> dist_op(0, 3);
    std::uniform_int_distribution<int> dist_idx(0, N - 1);

    std::vector<S> naive(N, 0);
    SumAddTree st(naive);

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
    SumAffineTree st(v);

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
    SumAffineTree st(naive);

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

static void test_assign_affine_together_on_sum_lazy() {
    using S = long long;
    using F = AssignAffineTag;

    std::vector<S> naive = {2, -1, 5, 3, 0, 4};
    SumAssignAffineTree st(naive);

    auto range_sum = [&](int l, int r) {
        S out = 0;
        for (int i = l; i < r; ++i) {
            out += naive[i];
        }
        return out;
    };

    st.apply_range(1, 5, F{true, 3, 2, -1});
    for (int i = 1; i < 5; ++i) {
        naive[i] = 2 * 3 - 1;
    }
    assert(st.prod(0, 6) == range_sum(0, 6));
    assert(st.prod(1, 5) == range_sum(1, 5));

    st.apply_range(2, 6, F{false, 0, -1, 4});
    for (int i = 2; i < 6; ++i) {
        naive[i] = -naive[i] + 4;
    }
    assert(st.prod(0, 6) == range_sum(0, 6));
    assert(st.prod(2, 6) == range_sum(2, 6));
}

static void test_assign_after_affine_on_sum_lazy() {
    using S = long long;
    using F = AssignAffineTag;

    std::vector<S> naive = {1, 2, 3, 4, 5, 6, 7};
    SumAssignAffineTree st(naive);

    auto range_sum = [&](int l, int r) {
        S out = 0;
        for (int i = l; i < r; ++i) {
            out += naive[i];
        }
        return out;
    };

    st.apply_range(1, 6, F{false, 0, 3, -2});
    for (int i = 1; i < 6; ++i) {
        naive[i] = 3 * naive[i] - 2;
    }

    st.apply_range(2, 5, F{true, 9, 1, 0});
    for (int i = 2; i < 5; ++i) {
        naive[i] = 9;
    }

    st.apply_range(3, 7, F{false, 0, -1, 1});
    for (int i = 3; i < 7; ++i) {
        naive[i] = -naive[i] + 1;
    }

    assert(st.prod(0, 7) == range_sum(0, 7));
    assert(st.prod(2, 6) == range_sum(2, 6));
    assert(st.get_point(3) == naive[3]);
}

static void stress_test_assign_affine_compositions_against_naive() {
    using S = long long;
    using F = AssignAffineTag;

    constexpr int N = 96;
    constexpr int STEPS = 9000;

    std::mt19937 rng(11235813u);
    std::uniform_int_distribution<int> dist_op(0, 4);
    std::uniform_int_distribution<int> dist_idx(0, N - 1);
    std::uniform_int_distribution<int> dist_val(-20, 20);
    std::uniform_int_distribution<int> dist_a(-1, 1);
    std::bernoulli_distribution coin(0.4);

    std::vector<S> naive(N, 0);
    SumAssignAffineTree st(naive);

    for (int step = 0; step < STEPS; ++step) {
        int l = dist_idx(rng);
        int r = dist_idx(rng);
        if (l > r) {
            std::swap(l, r);
        }
        ++r;

        const int op = dist_op(rng);
        if (op == 0 || op == 1) {
            F f{};
            f.has_assign = coin(rng);
            f.assign = dist_val(rng);
            f.a = dist_a(rng);
            f.b = dist_val(rng);

            st.apply_range(l, r, f);
            for (int i = l; i < r; ++i) {
                long long v = f.has_assign ? f.assign : naive[i];
                naive[i] = f.a * v + f.b;
            }
        } else if (op == 2) {
            S expect = 0;
            for (int i = l; i < r; ++i) {
                expect += naive[i];
            }
            assert(st.prod(l, r) == expect);
        } else if (op == 3) {
            const int p = dist_idx(rng);
            const S x = dist_val(rng);
            st.set_point(p, x);
            naive[p] = x;
        } else {
            const int p = dist_idx(rng);
            assert(st.get_point(p) == naive[p]);
        }
    }
}

static void test_ten_full_range_apply_then_single_prod_assign_affine() {
    using S = long long;
    using F = AssignAffineTag;

    std::vector<S> naive = {1, 3, 5, 7, 9, 11, 13, 15};
    SumAssignAffineTreeStatic st(naive);

    const std::vector<F> ops = {
        {false, 0, 1, 1},
        {false, 0, -1, 2},
        {true, 4, 1, 0},
        {false, 0, 1, -3},
        {false, 0, 0, 5},
        {false, 0, -1, 1},
        {true, -2, 2, 1},
        {false, 0, 1, 7},
        {false, 0, -1, 0},
        {false, 0, 1, 4},
    };

    const int n = static_cast<int>(naive.size());
    for (const F& f : ops) {
        st.apply_range(0, n, f);
        for (int i = 0; i < n; ++i) {
            long long v = f.has_assign ? f.assign : naive[i];
            naive[i] = f.a * v + f.b;
        }
    }

    long long expected = 0;
    for (long long x : naive) {
        expected += x;
    }
    assert(st.prod(0, n) == expected);
}

static void test_affine_negative_a_with_minmax_lazy() {
    using S = MinMax;
    using F = AffineTag;

    constexpr long long INF = std::numeric_limits<long long>::max() / 4;
    std::vector<long long> base = {7, -2, 5, 0, 9, -4, 3};
    std::vector<S> init;
    init.reserve(base.size());
    for (long long x : base) {
        init.push_back(minmax_point(x));
    }

    MinMaxAffineTree st(init);

    auto fold = [&](int l, int r) {
        long long mn = INF;
        long long mx = -INF;
        for (int i = l; i < r; ++i) {
            mn = std::min(mn, base[i]);
            mx = std::max(mx, base[i]);
        }
        return S{mn, mx};
    };

    st.apply_range(1, 6, F{-2, 3});
    for (int i = 1; i < 6; ++i) {
        base[i] = -2 * base[i] + 3;
    }

    st.apply_range(0, 4, F{3, -1});
    for (int i = 0; i < 4; ++i) {
        base[i] = 3 * base[i] - 1;
    }

    assert(st.prod(0, 7) == fold(0, 7));
    assert(st.prod(1, 6) == fold(1, 6));
    assert(st.prod(2, 4) == fold(2, 4));
}

static void stress_test_affine_minmax_with_negative_a_against_naive() {
    using S = MinMax;
    using F = AffineTag;

    constexpr int N = 90;
    constexpr int STEPS = 8000;
    constexpr long long INF = std::numeric_limits<long long>::max() / 4;

    std::mt19937 rng(42424242u);
    std::uniform_int_distribution<int> dist_op(0, 3);
    std::uniform_int_distribution<int> dist_idx(0, N - 1);
    std::uniform_int_distribution<int> dist_val(-50, 50);
    std::uniform_int_distribution<int> dist_a(-1, 1);

    std::vector<long long> naive(N, 0);
    std::vector<S> init(N, minmax_point(0));
    MinMaxAffineTree st(init);

    auto fold = [&](int l, int r) {
        long long mn = INF;
        long long mx = -INF;
        for (int i = l; i < r; ++i) {
            mn = std::min(mn, naive[i]);
            mx = std::max(mx, naive[i]);
        }
        return S{mn, mx};
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
            const F f{dist_a(rng), dist_val(rng)};
            st.apply_range(l, r, f);
            for (int i = l; i < r; ++i) {
                naive[i] = f.a * naive[i] + f.b;
            }
        } else if (op == 1) {
            assert(st.prod(l, r) == fold(l, r));
        } else if (op == 2) {
            const int p = dist_idx(rng);
            const long long x = dist_val(rng);
            st.set_point(p, minmax_point(x));
            naive[p] = x;
        } else {
            const int p = dist_idx(rng);
            const S got = st.get_point(p);
            assert(got.mn == naive[p] && got.mx == naive[p]);
        }
    }
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

    BitwiseAffineTree st(init);

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

    BitwiseAffineTree st(init);

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

static void test_xor_lazy_requested_patterns() {
    using S = std::uint32_t;

    std::vector<S> naive = {3, 7, 1, 4, 2, 8, 5, 9, 6, 10, 12, 14};
    XorLazyTree st(naive);

    auto fold = [&](int l, int r) {
        S out = 0;
        for (int i = l; i < r; ++i) {
            out ^= naive[i];
        }
        return out;
    };

    const int n = static_cast<int>(naive.size());
    const int k = 5;
    st.apply_range(0, n, (1u << k));
    for (int i = 0; i < n; ++i) {
        naive[i] ^= (1u << k);
    }
    st.apply_range(0, n, (1u << k));
    for (int i = 0; i < n; ++i) {
        naive[i] ^= (1u << k);
    }
    assert(st.prod(0, n) == fold(0, n));

    const int l = 2;
    const int mid = 7;
    const int r = 11;
    const S m = 0x35u;
    st.apply_range(l, r, m);
    for (int i = l; i < r; ++i) {
        naive[i] ^= m;
    }
    st.apply_range(l, mid, m);
    for (int i = l; i < mid; ++i) {
        naive[i] ^= m;
    }

    assert(st.prod(0, n) == fold(0, n));
    assert(st.prod(l, r) == fold(l, r));
    assert(st.prod(l, mid) == fold(l, mid));
    assert(st.prod(mid, r) == fold(mid, r));

    std::vector<S> full_masks = {1u, 3u, 7u, 15u, 31u, 63u, 127u, 255u, 511u, 1023u};
    for (S mask : full_masks) {
        st.apply_range(0, n, mask);
        for (int i = 0; i < n; ++i) {
            naive[i] ^= mask;
        }
    }
    assert(st.prod(0, n) == fold(0, n));
}

static void stress_test_xor_lazy_against_naive() {
    using S = std::uint32_t;

    constexpr int N = 128;
    constexpr int STEPS = 8000;

    std::mt19937 rng(1357911u);
    std::uniform_int_distribution<int> dist_op(0, 3);
    std::uniform_int_distribution<int> dist_idx(0, N - 1);
    std::uniform_int_distribution<S> dist_mask(0u, 0x3FFu);
    std::uniform_int_distribution<S> dist_val(0u, 0x7FFu);

    std::vector<S> naive(N, 0);
    for (int i = 0; i < N; ++i) {
        naive[i] = dist_val(rng);
    }
    XorLazyTree st(naive);

    auto fold = [&](int l, int r) {
        S out = 0;
        for (int i = l; i < r; ++i) {
            out ^= naive[i];
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
            const S m = dist_mask(rng);
            st.apply_range(l, r, m);
            for (int i = l; i < r; ++i) {
                naive[i] ^= m;
            }
        } else if (op == 1) {
            assert(st.prod(l, r) == fold(l, r));
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
}

static void test_matrix_product_with_assign_and_affine_lazy() {
    using S = Mat2;
    using F = MatConjAssignTag;

    constexpr long long MOD = 1000000007LL;
    auto norm = [&](long long x) {
        x %= MOD;
        if (x < 0) {
            x += MOD;
        }
        return x;
    };
    auto mul = [&](const S& a, const S& b) {
        return S{
            norm(a.a00 * b.a00 + a.a01 * b.a10),
            norm(a.a00 * b.a01 + a.a01 * b.a11),
            norm(a.a10 * b.a00 + a.a11 * b.a10),
            norm(a.a10 * b.a01 + a.a11 * b.a11),
        };
    };
    auto apply_point = [&](const F& f, const S& x) {
        S v = f.has_assign ? f.assign : x;
        v = mul(f.l, v);
        v = mul(v, f.r);
        return v;
    };

    std::vector<S> naive = {
        {1, 2, 0, 1},
        {2, 0, 1, 3},
        {3, 1, 4, 1},
        {1, 1, 1, 0},
        {0, 1, 2, 1},
        {2, 3, 1, 2},
    };
    for (S& x : naive) {
        x = S{norm(x.a00), norm(x.a01), norm(x.a10), norm(x.a11)};
    }

    MatrixAssignAffineTree st(naive);

    auto upper_conj = [&](long long t) {
        S l{1, norm(t), 0, 1};
        S r{1, norm(-t), 0, 1};
        return F{false, S{0, 0, 0, 0}, l, r};
    };
    auto lower_conj = [&](long long t) {
        S l{1, 0, norm(t), 1};
        S r{1, 0, norm(-t), 1};
        return F{false, S{0, 0, 0, 0}, l, r};
    };
    auto assign_only = [&](const S& c) {
        return F{true, c, S{1, 0, 0, 1}, S{1, 0, 0, 1}};
    };
    auto assign_with_conj = [&](const S& c, long long t) {
        const F g = upper_conj(t);
        return F{true, c, g.l, g.r};
    };

    auto fold = [&](int l, int r) {
        S out{1, 0, 0, 1};
        for (int i = l; i < r; ++i) {
            out = mul(out, naive[i]);
        }
        return out;
    };
    auto apply_naive = [&](int l, int r, const F& f) {
        for (int i = l; i < r; ++i) {
            naive[i] = apply_point(f, naive[i]);
        }
    };

    F f1 = upper_conj(2);
    st.apply_range(1, 5, f1);
    apply_naive(1, 5, f1);

    F f2 = assign_only(S{2, 1, 1, 1});
    st.apply_range(2, 4, f2);
    apply_naive(2, 4, f2);

    F f3 = assign_with_conj(S{3, 0, 1, 2}, 1);
    st.apply_range(0, 3, f3);
    apply_naive(0, 3, f3);

    F f4 = lower_conj(3);
    st.apply_range(0, 6, f4);
    apply_naive(0, 6, f4);

    assert(st.prod(0, 6) == fold(0, 6));
    assert(st.prod(1, 5) == fold(1, 5));
    assert(st.prod(2, 3) == fold(2, 3));
}

static void stress_test_matrix_assign_affine_compositions_against_naive() {
    using S = Mat2;
    using F = MatConjAssignTag;

    constexpr long long MOD = 1000000007LL;
    constexpr int N = 40;
    constexpr int STEPS = 7000;

    auto norm = [&](long long x) {
        x %= MOD;
        if (x < 0) {
            x += MOD;
        }
        return x;
    };
    auto mul = [&](const S& a, const S& b) {
        return S{
            norm(a.a00 * b.a00 + a.a01 * b.a10),
            norm(a.a00 * b.a01 + a.a01 * b.a11),
            norm(a.a10 * b.a00 + a.a11 * b.a10),
            norm(a.a10 * b.a01 + a.a11 * b.a11),
        };
    };
    auto apply_point = [&](const F& f, const S& x) {
        S v = f.has_assign ? f.assign : x;
        v = mul(f.l, v);
        v = mul(v, f.r);
        return v;
    };
    auto conj_upper = [&](long long t) {
        const long long tt = norm(t);
        S l{1, tt, 0, 1};
        S r{1, norm(-tt), 0, 1};
        return F{false, S{0, 0, 0, 0}, l, r};
    };
    auto id_tag = [&]() {
        return F{false, S{0, 0, 0, 0}, S{1, 0, 0, 1}, S{1, 0, 0, 1}};
    };

    std::mt19937 rng(77773331u);
    std::uniform_int_distribution<int> dist_op(0, 3);
    std::uniform_int_distribution<int> dist_idx(0, N - 1);
    std::uniform_int_distribution<int> dist_small(-3, 3);
    std::uniform_int_distribution<int> dist_t(0, 8);
    std::bernoulli_distribution coin(0.5);

    std::vector<S> naive(N);
    for (int i = 0; i < N; ++i) {
        naive[i] = S{norm(1), norm(dist_small(rng)), norm(dist_small(rng)), norm(1)};
    }

    MatrixAssignAffineTree st(naive);

    auto fold = [&](int l, int r) {
        S out{1, 0, 0, 1};
        for (int i = l; i < r; ++i) {
            out = mul(out, naive[i]);
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
            F f = id_tag();
            if (coin(rng)) {
                f = conj_upper(dist_t(rng));
            } else {
                f.has_assign = true;
                f.assign = S{
                    norm(dist_small(rng)),
                    norm(dist_small(rng)),
                    norm(dist_small(rng)),
                    norm(dist_small(rng)),
                };
                if (coin(rng)) {
                    const F c = conj_upper(dist_t(rng));
                    f.l = c.l;
                    f.r = c.r;
                }
            }

            st.apply_range(l, r, f);
            for (int i = l; i < r; ++i) {
                naive[i] = apply_point(f, naive[i]);
            }
        } else if (op == 1) {
            assert(st.prod(l, r) == fold(l, r));
        } else if (op == 2) {
            const int p = dist_idx(rng);
            S x{
                norm(dist_small(rng)),
                norm(dist_small(rng)),
                norm(dist_small(rng)),
                norm(dist_small(rng)),
            };
            st.set_point(p, x);
            naive[p] = x;
        } else {
            const int p = dist_idx(rng);
            assert(st.get_point(p) == naive[p]);
        }
    }
}

static void test_polynomial_index_update_sequence_requested() {
    using S = PolyNode;
    using F = PolyTag;

    const int n = 8;
    std::vector<long long> naive = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<S> init;
    init.reserve(n);
    for (int i = 0; i < n; ++i) {
        init.push_back(poly_point(naive[i], i));
    }

    PolyTree st(init);

    auto apply_naive = [&](int l, int r, const F& f) {
        for (int i = l; i < r; ++i) {
            naive[i] = f.a * naive[i] + f.c0 + f.c1 * i + f.c2 * i * i;
        }
    };
    auto sum_naive = [&](int l, int r) {
        long long out = 0;
        for (int i = l; i < r; ++i) {
            out += naive[i];
        }
        return out;
    };

    st.apply_range(0, n, F{1, 1, 2, 3});
    apply_naive(0, n, F{1, 1, 2, 3});

    st.apply_range(2, n - 1, F{-2, 5, -1, 0});
    apply_naive(2, n - 1, F{-2, 5, -1, 0});

    st.apply_range(1, 4, F{1, 7, 0, 1});
    apply_naive(1, 4, F{1, 7, 0, 1});

    st.apply_range(0, n, F{3, 1, 0, 0});
    apply_naive(0, n, F{3, 1, 0, 0});

    st.apply_range(0, n, F{1, 0, 2, 0});
    apply_naive(0, n, F{1, 0, 2, 0});

    st.apply_range(0, n, F{-1, 4, 0, 0});
    apply_naive(0, n, F{-1, 4, 0, 0});

    st.set_point(3, poly_point(10, 3));
    naive[3] = 10;

    assert(st.prod(0, n).sum == sum_naive(0, n));
    assert(st.prod(2, n - 1).sum == sum_naive(2, n - 1));
    assert(st.get_point(3).sum == naive[3]);
}

static void stress_test_polynomial_index_updates_against_naive() {
    using S = PolyNode;
    using F = PolyTag;

    constexpr int N = 64;
    constexpr int STEPS = 8000;

    std::mt19937 rng(918273645u);
    std::uniform_int_distribution<int> dist_op(0, 3);
    std::uniform_int_distribution<int> dist_idx(0, N - 1);
    std::uniform_int_distribution<int> dist_a(-1, 1);
    std::uniform_int_distribution<int> dist_c(-5, 5);
    std::uniform_int_distribution<int> dist_v(-40, 40);

    std::vector<long long> naive(N, 0);
    std::vector<S> init;
    init.reserve(N);
    for (int i = 0; i < N; ++i) {
        init.push_back(poly_point(0, i));
    }

    PolyTree st(init);

    auto sum_naive = [&](int l, int r) {
        long long out = 0;
        for (int i = l; i < r; ++i) {
            out += naive[i];
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
            const F f{dist_a(rng), dist_c(rng), dist_c(rng), dist_c(rng)};
            st.apply_range(l, r, f);
            for (int i = l; i < r; ++i) {
                naive[i] = f.a * naive[i] + f.c0 + f.c1 * i + f.c2 * i * i;
            }
        } else if (op == 1) {
            assert(st.prod(l, r).sum == sum_naive(l, r));
        } else if (op == 2) {
            const int p = dist_idx(rng);
            const long long x = dist_v(rng);
            st.set_point(p, poly_point(x, p));
            naive[p] = x;
        } else {
            const int p = dist_idx(rng);
            assert(st.get_point(p).sum == naive[p]);
        }
    }
}

static void test_monoid_sum_tree() {
    run_monoid_checks<long long, op_sum_ll, e_sum_ll>(
        {3, -1, 4, 1, 5, -9, 2, 6},
        {{0, 10}, {5, 7}, {7, -3}});
}

static void test_monoid_min_tree() {
    run_monoid_checks<long long, op_min_ll, e_min_ll>(
        {7, 2, 9, 4, 8, 1, 6},
        {{3, -5}, {1, 11}, {5, 0}});
}

static void test_monoid_max_tree() {
    run_monoid_checks<long long, op_max_ll, e_max_ll>(
        {7, 2, 9, 4, 8, 1, 6},
        {{3, -5}, {1, 11}, {5, 0}});
}

static void test_monoid_gcd_tree() {
    run_monoid_checks<long long, op_gcd_ll, e_sum_ll>(
        {12, 18, 30, 42, 54, 66},
        {{0, 24}, {2, 45}, {5, 81}});
}

static void test_monoid_xor_tree() {
    run_monoid_checks<std::uint32_t, op_xor_u32, e_xor_u32>(
        {0x1u, 0x3u, 0x7u, 0xFu, 0x10u, 0x55u},
        {{1, 0xABu}, {3, 0x33u}, {4, 0x0u}});
}

static void test_monoid_or_tree() {
    run_monoid_checks<std::uint32_t, op_or_u32, e_or_u32>(
        {0x1u, 0x2u, 0x4u, 0x8u, 0x10u},
        {{0, 0x40u}, {2, 0x80u}, {4, 0x0u}});
}

static void test_monoid_and_tree() {
    run_monoid_checks<std::uint32_t, op_and_u32, e_and_u32>(
        {0xFFu, 0xF0u, 0xCCu, 0xAAu, 0x0Fu},
        {{0, 0x3Fu}, {1, 0x55u}, {4, 0xF3u}});
}

static void test_monoid_lcm_tree() {
    run_monoid_checks<std::uint64_t, op_lcm_u64, e_lcm_u64>(
        {2ull, 3ull, 4ull, 5ull, 6ull},
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
    run_monoid_checks<S, op_mat2_plain, e_mat2_plain>(
        init,
        {{1, Mat2{2, 0, 1, 2}}, {3, Mat2{1, 1, 0, 1}}});
}

static void test_monoid_matrix_product_tree_3x3() {
    using S = Mat3;
    const std::vector<S> init = {
        {{{{1, 2, 0}, {0, 1, 1}, {2, 0, 1}}}},
        {{{{2, 1, 3}, {1, 0, 2}, {0, 1, 1}}}},
        {{{{1, 0, 1}, {3, 1, 0}, {2, 2, 1}}}},
        {{{{0, 1, 2}, {1, 1, 0}, {4, 0, 1}}}},
    };
    run_monoid_checks<S, op_mat3_plain, e_mat3_plain>(
        init,
        {
            {1, S{{{{1, 1, 0}, {0, 1, 1}, {2, 0, 1}}}}},
            {3, S{{{{2, 0, 1}, {1, 3, 0}, {0, 1, 1}}}}},
        });
}

static void test_monoid_pair_tree() {
    run_monoid_checks<SumMax, op_summax, e_summax>(
        {
            {3, 3},
            {1, 1},
            {4, 4},
            {1, 1},
            {5, 5},
            {9, 9},
        },
        {{1, SumMax{10, 10}}, {4, SumMax{-2, -2}}, {5, SumMax{7, 7}}});
}

int main() {
    test_sum_with_add_lazy();
    test_sum_with_affine_lazy();
    test_assign_affine_together_on_sum_lazy();
    test_assign_after_affine_on_sum_lazy();
    test_ten_full_range_apply_then_single_prod_assign_affine();
    test_affine_negative_a_with_minmax_lazy();
    test_sum_with_bitwise_affine_lazy();
    test_max_with_assign_lazy();
    test_xor_lazy_requested_patterns();
    test_matrix_product_with_assign_and_affine_lazy();
    test_polynomial_index_update_sequence_requested();

    stress_test_sum_add_lazy_against_naive();
    stress_test_sum_affine_lazy_against_naive();
    stress_test_assign_affine_compositions_against_naive();
    stress_test_affine_minmax_with_negative_a_against_naive();
    stress_test_sum_bitwise_affine_lazy_against_naive();
    stress_test_xor_lazy_against_naive();
    stress_test_matrix_assign_affine_compositions_against_naive();
    stress_test_polynomial_index_updates_against_naive();

    test_monoid_sum_tree();
    test_monoid_min_tree();
    test_monoid_max_tree();
    test_monoid_gcd_tree();
    test_monoid_xor_tree();
    test_monoid_or_tree();
    test_monoid_and_tree();
    test_monoid_lcm_tree();
    test_monoid_matrix_product_tree();
    test_monoid_matrix_product_tree_3x3();
    test_monoid_pair_tree();

    std::cout << "All LazySegTree tests passed.\n";
    return 0;
}
