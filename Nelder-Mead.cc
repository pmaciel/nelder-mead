#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>


using function_t = std::function<double(const std::vector<double>)>;


struct matrix_row_t : std::vector<double> {
    using vector<double>::vector;
    using vector<double>::operator=;

    friend matrix_row_t operator*(const matrix_row_t& a, double b) {
        matrix_row_t c(a);
        for (auto& e : c) {
            e *= b;
        }
        return c;
    }

    friend matrix_row_t operator+(const matrix_row_t& a, const matrix_row_t& b) {
        assert(a.size() == b.size());
        matrix_row_t c(a);
        for (size_t i = 0; i < c.size(); ++i) {
            c[i] += b[i];
        }
        return c;
    }
};


struct matrix_t : protected std::vector<matrix_row_t> {
    matrix_t(size_t Ni, size_t Nj) : vector<matrix_row_t>(Ni, matrix_row_t(Nj, 0)) {}

    matrix_row_t::value_type operator()(size_t i, size_t j) const { return at(i).at(j); }
    matrix_row_t::value_type& operator()(size_t i, size_t j) { return at(i).at(j); }

    const matrix_row_t& row(size_t i) const { return at(i); }
    matrix_row_t& row(size_t i) { return at(i); }

    size_t rows() const { return size(); }
    size_t cols() const { return empty() ? 0 : front().size(); }

    using vector<matrix_row_t>::begin;
    using vector<matrix_row_t>::end;
    using vector<matrix_row_t>::at;
};


struct extrema_t {
    explicit extrema_t(size_t _high, size_t _nexthigh, size_t _low) : high(_high), nexthigh(_nexthigh), low(_low) {}
    explicit extrema_t(const std::vector<double>& fx) {
        auto b   = fx[0] > fx[1];
        high     = b ? 0 : 1;
        nexthigh = b ? 1 : 0;
        low      = b ? 1 : 0;

        for (size_t i = 2; i < fx.size(); ++i) {
            auto fi = fx[i];

            if (fi <= fx[low]) {
                low = i;
            }
            else if (fi > fx[high]) {
                nexthigh = high;
                high     = i;
            }
            else if (fi > fx[nexthigh]) {
                nexthigh = i;
            }
        }
    }

    size_t high     = 0;  // worst (f(x))
    size_t nexthigh = 0;  // 2nd worse (f(x))
    size_t low      = 0;  // best (f(x))
};


class NelderMead {
public:
    NelderMead(size_t dim, const function_t& f) : dim_(dim), f_(f) {}

    double minimize(std::vector<double>& x, double tol, size_t maxcount, std::ostream& out) {
        extrema_t ex{0, 0, 0};
        double eps = epsilon();

        // initial simplex: ndim+1 points to evaluate, and perturb starting points
        matrix_t S(dim_ + 1, dim_);
        for (size_t i = 0; i < S.rows(); ++i) {
            S.row(i) = x;
            if (i < S.cols()) {
                S(i, i) *= 1.01;
            }
        }

        // evaluate at the simplex points
        std::vector<double> fx;
        fx.reserve(S.rows());
        for (const auto& x : S) {
            fx.emplace_back(f_(x));
        }

        for (size_t count = 1; count < maxcount; ++count) {
            // find extremes at the simplex: high (worst), nexthigh (2nd worse) and low (best)
            ex = extrema_t(fx);

            // find optimization direction (vOpt) from the worst (high) point to the centroid (vMid) of all other
            // points
            matrix_row_t vMid(S.cols(), 0.);
            for (size_t i = 0; i < S.rows(); ++i) {
                if (i != ex.high) {
                    vMid = vMid + S.row(i) * (1. / double(S.cols()));
                }
            }

            auto vOpt = S.row(ex.high) + (vMid * -1.);

            auto& fhigh     = fx[ex.high];
            auto& fnexthigh = fx[ex.nexthigh];
            auto& flow      = fx[ex.low];

            auto delta    = std::abs(fhigh - flow);
            auto accuracy = std::abs(fhigh) + std::abs(flow);
            if (delta < accuracy * tol + eps) {
                break;
            }

            update(S, vMid, vOpt, ex, -1., fhigh);

            if (fhigh < flow) {
                update(S, vMid, vOpt, ex, -2., fhigh);
            }
            else if (fhigh >= fnexthigh) {
                if (!update(S, vMid, vOpt, ex, 0.5, fhigh)) {
                    // contract existing simplex, hoping to achieve an update

                    for (size_t i = 0; i < S.rows(); ++i) {
                        if (i != ex.low) {
                            S.row(i) = (S.row(ex.low) + S.row(i)) * 0.5;
                            fx[i]    = f_(S.row(i));
                        }
                    }
                }
            }

            out << "iter: " << count << ",\ty: " << fx[ex.low] << ",\tx: ";
            const auto* sep = "";
            for (size_t i = 0; i < S.cols(); i++) {
                out << sep << S(i, i);
                sep = ",\t";
            }
            out << std::endl;
        }

        // return parameter values at the minimum (and the value of the minimum)
        x = S.row(ex.low);
        return fx[ex.low];
    }

private:
    size_t dim_;
    const function_t& f_;

    double epsilon() {
        int p = 1;
        for (double a, b;; ++p) {
            a = 1. + std::pow(2, -1 * p);
            b = a - 1.;
            if (b <= 0) {
                break;
            }
        }

        return std::pow(2, -1 * (p - 1));
    }

    bool update(matrix_t& S, const matrix_row_t& vMid, const matrix_row_t& vOpt, const extrema_t& ex, double scale,
                double& fmax) {
        // update simplex if a new minimum is found, according to "scale"

        auto x  = vMid + (vOpt * scale);
        auto fx = f_(x);

        if (fmax > fx) {
            fmax           = fx;
            S.row(ex.high) = x;
            return true;
        }

        return false;
    }
};


int main() {
    struct {
        const char* name;
        std::vector<double> x0;
        function_t fx;
    } tests[]{
        {"Rosenbrock function",
         {-1.2, 1.},
         [](const std::vector<double>& x) {
             return 100 * std::pow((x[1] - x[0] * x[0]), 2) + std::pow((1 - x[0]), 2);
         }},
        {"Fletcher-Powell's helical valley function",
         {-1., 2., 1.},
         [](const std::vector<double>& x) {
             auto theta_term = x[0] > 0 ? (5 / M_PI) * atan(x[1] / x[0]) : M_PI + (5 / M_PI) * atan(x[1] / x[0]);
             return 100 * std::pow((x[2] - theta_term), 2) + std::pow(sqrt(x[0] * x[0] + x[1] * x[1]), 2) + x[2] * x[2];
         }},
        {"Powell's quartic function",
         {3., -1., 0., 1.},
         [](const std::vector<double>& x) {
             return std::pow((x[0] + 10 * x[1]), 2) + 5 * std::pow((x[2] - x[3]), 2) + std::pow((x[1] - 2 * x[2]), 4) +
                    10 * std::pow((x[0] - x[3]), 4);
         }},
    };

    for (const auto& test : tests) {
        std::cout << test.name << std::endl;

        NelderMead nm(test.x0.size(), test.fx);
        auto x = test.x0;
        nm.minimize(x, 1.e-8, 5000, std::cout);
    }

    return 0;
}
