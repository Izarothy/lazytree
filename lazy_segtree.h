#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>
#include <vector>

namespace lazy_segtree_detail {

constexpr int ceil_pow2(int n) {
    int out = 1;
    while (out < n) {
        out <<= 1;
    }
    return out;
}

template <class T, int Capacity, bool Dynamic>
class Storage;

template <class T, int Capacity>
class Storage<T, Capacity, true> {
public:
    void resize(int n, const T& value) {
        data_.assign(n, value);
    }

    T& operator[](int i) {
        return data_[i];
    }

    const T& operator[](int i) const {
        return data_[i];
    }

private:
    std::vector<T> data_;
};

template <class T, int Capacity>
class Storage<T, Capacity, false> {
public:
    void resize(int n, const T& value) {
        assert(0 <= n && n <= Capacity);
        used_ = n;
        std::fill(data_.begin(), data_.begin() + used_, value);
    }

    T& operator[](int i) {
        assert(0 <= i && i < used_);
        return data_[i];
    }

    const T& operator[](int i) const {
        assert(0 <= i && i < used_);
        return data_[i];
    }

private:
    std::array<T, Capacity> data_{};
    int used_ = 0;
};

} // namespace lazy_segtree_detail

template <
    class S,
    S (*op)(S, S),
    S (*e)(),
    class F,
    S (*mapping)(F, S, int),
    F (*composition)(F, F),
    F (*id)(),
    int MAX_N = 0>
class LazySegTree {
    static_assert(MAX_N >= 0, "MAX_N must be >= 0");

    static constexpr bool kDynamic = (MAX_N == 0);
    static constexpr int kLeafCap = kDynamic ? 1 : lazy_segtree_detail::ceil_pow2(MAX_N);
    static constexpr int kNodeCap = kDynamic ? 1 : (2 * kLeafCap);

    using NodeStorage = std::conditional_t<
        kDynamic,
        lazy_segtree_detail::Storage<S, kNodeCap, true>,
        lazy_segtree_detail::Storage<S, kNodeCap, false>>;
    using LazyStorage = std::conditional_t<
        kDynamic,
        lazy_segtree_detail::Storage<F, kLeafCap, true>,
        lazy_segtree_detail::Storage<F, kLeafCap, false>>;

public:
    int n;
    int size;
    int log;

    LazySegTree()
        : n(0), size(1), log(0) {
        init_storage();
    }

    explicit LazySegTree(int n_)
        : n(0), size(1), log(0) {
        build_empty(n_);
    }

    explicit LazySegTree(const std::vector<S>& v)
        : n(0), size(1), log(0) {
        build_from_vector(v);
    }

    void set_point(int p, S x) {
        assert(0 <= p && p < n);
        p += size;
        for (int i = log; i >= 1; --i) {
            push(p >> i, 1 << i);
        }
        d[p] = x;
        for (int i = 1; i <= log; ++i) {
            pull(p >> i);
        }
    }

    S get_point(int p) {
        assert(0 <= p && p < n);
        p += size;
        for (int i = log; i >= 1; --i) {
            push(p >> i, 1 << i);
        }
        return d[p];
    }

    void apply_range(int l, int r, F f) {
        assert(0 <= l && l <= r && r <= n);
        if (l == r) {
            return;
        }

        l += size;
        r += size;

        for (int i = log; i >= 1; --i) {
            if (((l >> i) << i) != l) {
                push(l >> i, 1 << i);
            }
            if (((r >> i) << i) != r) {
                push((r - 1) >> i, 1 << i);
            }
        }

        const int l0 = l;
        const int r0 = r;
        int len = 1;
        while (l < r) {
            if (l & 1) {
                all_apply(l++, f, len);
            }
            if (r & 1) {
                all_apply(--r, f, len);
            }
            l >>= 1;
            r >>= 1;
            len <<= 1;
        }

        for (int i = 1; i <= log; ++i) {
            if (((l0 >> i) << i) != l0) {
                pull(l0 >> i);
            }
            if (((r0 >> i) << i) != r0) {
                pull((r0 - 1) >> i);
            }
        }
    }

    S prod(int l, int r) {
        assert(0 <= l && l <= r && r <= n);
        if (l == r) {
            return e();
        }

        l += size;
        r += size;
        for (int i = log; i >= 1; --i) {
            if (((l >> i) << i) != l) {
                push(l >> i, 1 << i);
            }
            if (((r >> i) << i) != r) {
                push((r - 1) >> i, 1 << i);
            }
        }

        S sml = e();
        S smr = e();
        while (l < r) {
            if (l & 1) {
                sml = op(sml, d[l++]);
            }
            if (r & 1) {
                smr = op(d[--r], smr);
            }
            l >>= 1;
            r >>= 1;
        }

        return op(sml, smr);
    }

    S all_prod() const {
        return d[1];
    }

private:
    NodeStorage d;
    LazyStorage lz;

    void check_static_bound(int n_) const {
        if constexpr (!kDynamic) {
            assert(n_ <= MAX_N);
        }
    }

    void init_storage() {
        d.resize(2, e());
        lz.resize(1, id());
    }

    void build_empty(int n_) {
        assert(n_ >= 0);
        check_static_bound(n_);
        n = n_;
        size = 1;
        log = 0;
        while (size < n) {
            size <<= 1;
            ++log;
        }
        d.resize(2 * size, e());
        lz.resize(size, id());
        for (int k = size - 1; k >= 1; --k) {
            pull(k);
        }
    }

    void build_from_vector(const std::vector<S>& v) {
        build_empty(static_cast<int>(v.size()));
        for (int i = 0; i < n; ++i) {
            d[size + i] = v[i];
        }
        for (int k = size - 1; k >= 1; --k) {
            pull(k);
        }
    }

    void pull(int k) {
        d[k] = op(d[2 * k], d[2 * k + 1]);
    }

    void push(int k, int len) {
        if (k >= size) {
            return;
        }
        assert(seglen(k) == len);
        const int half = len >> 1;
        all_apply(2 * k, lz[k], half);
        all_apply(2 * k + 1, lz[k], half);
        lz[k] = id();
    }

    void all_apply(int k, F f, int len) {
        d[k] = mapping(f, d[k], len);
        if (k < size) {
            lz[k] = composition(f, lz[k]);
        }
    }

    int seglen(int k) const {
        int depth = 0;
        while (k > 1) {
            k >>= 1;
            ++depth;
        }
        return size >> depth;
    }
};
