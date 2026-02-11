#pragma once

#include <cassert>
#include <functional>
#include <utility>
#include <vector>

template <class S, class F>
class LazySegTree {
public:
    using Op = std::function<S(const S&, const S&)>;
    using E = std::function<S()>;
    using Apply = std::function<S(const F&, const S&, int)>;
    using Compose = std::function<F(const F&, const F&)>;
    using Id = std::function<F()>;

    int n;
    int size;
    int log;

    Op op;
    E e;
    Apply apply;
    Compose compose;
    Id id;

    std::vector<S> d;
    std::vector<F> lz;

    LazySegTree()
        : n(0), size(1), log(0), d(), lz() {}

    LazySegTree(int n_, Op op_, E e_, Apply apply_, Compose compose_, Id id_)
        : n(n_), size(1), log(0), op(std::move(op_)), e(std::move(e_)),
          apply(std::move(apply_)), compose(std::move(compose_)), id(std::move(id_)) {
        assert(n >= 0);
        while (size < n) {
            size <<= 1;
            ++log;
        }
        d.assign(2 * size, e());
        lz.assign(size, id());
        for (int k = size - 1; k >= 1; --k) {
            pull(k);
        }
    }

    LazySegTree(const std::vector<S>& v, Op op_, E e_, Apply apply_, Compose compose_, Id id_)
        : n(static_cast<int>(v.size())), size(1), log(0), op(std::move(op_)), e(std::move(e_)),
          apply(std::move(apply_)), compose(std::move(compose_)), id(std::move(id_)) {
        while (size < n) {
            size <<= 1;
            ++log;
        }
        d.assign(2 * size, e());
        lz.assign(size, id());
        for (int i = 0; i < n; ++i) {
            d[size + i] = v[i];
        }
        for (int k = size - 1; k >= 1; --k) {
            pull(k);
        }
    }

    void set_point(int p, const S& x) {
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

    void apply_range(int l, int r, const F& f) {
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

        int l0 = l;
        int r0 = r;
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

    void all_apply(int k, const F& f, int len) {
        d[k] = apply(f, d[k], len);
        if (k < size) {
            lz[k] = compose(f, lz[k]);
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
