#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>


namespace optim {


class Method {
public:
    using function_t = std::function<double(const std::vector<double>)>;

    Method()              = default;
    Method(const Method&) = delete;
    Method(Method&&)      = default;

    void operator=(const Method&) = delete;
    Method& operator=(Method&&) = default;

    virtual ~Method()                      = default;
    virtual double minimize(const function_t& f, std::vector<double>& x, double tol, size_t maxcount,
                            std::ostream&) = 0;
};


class NelderMead : public Method {
private:
    struct matrix_row_t : std::vector<double> {
        using vector<double>::vector;
        using vector<double>::operator=;

        friend matrix_row_t operator*(double a, const matrix_row_t& b) {
            matrix_row_t c(b);
            for (auto& e : c) {
                e *= a;
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

        friend matrix_row_t operator-(const matrix_row_t& a, const matrix_row_t& b) {
            assert(a.size() == b.size());
            matrix_row_t c(a);
            for (size_t i = 0; i < c.size(); ++i) {
                c[i] -= b[i];
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
        explicit extrema_t() = default;
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

    const double alpha_;
    const double gamma_;
    const double rho_;
    const double sigma_;

public:
    NelderMead(double alpha = 1., double gamma = 2., double rho = 0.5, double sigma = 0.5) :
        alpha_(alpha), gamma_(gamma), rho_(rho), sigma_(sigma) {}

    double minimize(const function_t& f, std::vector<double>& x, double tol, size_t maxcount,
                    std::ostream& out) override {
        extrema_t ex;

        auto eps = []() -> double {
            int p = 1;
            for (double a, b;; ++p) {
                a = 1. + std::pow(2, -1 * p);
                b = a - 1.;
                if (b <= 0) {
                    break;
                }
            }

            return std::pow(2, -1 * (p - 1));
        }();

        // initial simplex: ndim+1 points to evaluate, and perturb starting points
        const auto dim = x.size();
        matrix_t X(dim + 1, dim);
        for (size_t i = 0; i < X.rows(); ++i) {
            X.row(i) = x;
            if (i < X.cols()) {
                X(i, i) *= 1.01;
            }
        }

        // evaluate at the simplex points
        std::vector<double> fx;
        fx.reserve(X.rows());
        for (const auto& x : X) {
            fx.emplace_back(f(x));
        }

        for (size_t count = 1; count < maxcount; ++count) {
            // find extremes at the simplex: high (worst), nexthigh (2nd worse) and low (best)
            ex = extrema_t(fx);

            // find optimization direction (vOpt) from the worst (high) point to the centroid (vMid) of all other
            // points
            matrix_row_t vMid(X.cols(), 0.);
            for (size_t i = 0; i < X.rows(); ++i) {
                if (i != ex.high) {
                    vMid = vMid + (1. / double(X.cols()) * X.row(i));
                }
            }

            auto vOpt = X.row(ex.high) - vMid;

            auto& fhigh     = fx[ex.high];
            auto& fnexthigh = fx[ex.nexthigh];
            auto& flow      = fx[ex.low];

            auto delta    = std::abs(fhigh - flow);
            auto accuracy = std::abs(fhigh) + std::abs(flow);
            if (delta < accuracy * tol + eps) {
                break;
            }

            {
                auto x_ = vMid - alpha_ * vOpt;
                auto f_ = f(x_);

                // update simplex if a new minimum is found
                if (fhigh > f_) {
                    fhigh          = f_;
                    X.row(ex.high) = x_;
                }
            }

            if (fhigh < flow) {
                auto x_ = vMid - gamma_ * vOpt;
                auto f_ = f(x_);

                // update simplex if a new minimum is found
                if (fhigh > f_) {
                    fhigh          = f_;
                    X.row(ex.high) = x_;
                }
            }
            else if (fhigh >= fnexthigh) {
                auto x_ = vMid + rho_ * vOpt;
                auto f_ = f(x_);

                // update simplex if a new minimum is found
                if (fhigh > f_) {
                    fhigh          = f_;
                    X.row(ex.high) = x_;
                }
                else {

                    // contract existing simplex, hoping to achieve an update
                    for (size_t i = 0; i < X.rows(); ++i) {
                        if (i != ex.low) {
                            X.row(i) = sigma_ * (X.row(ex.low) + X.row(i));
                            fx[i]    = f(X.row(i));
                        }
                    }
                }
            }

            out << "iter: " << count << ",\ty: " << fx[ex.low] << ",\tx: ";
            const auto* sep = "";
            for (size_t i = 0; i < X.cols(); i++) {
                out << sep << X(i, i);
                sep = ",\t";
            }
            out << std::endl;
        }

        // return parameter values at the minimum (and the value of the minimum)
        x = X.row(ex.low);
        return fx[ex.low];
    }
};


}  // namespace optim


int main() {
    struct {
        const char* name;
        std::vector<double> x0;
        optim::Method::function_t f;
    } tests[]{
        {"Rosenbrock function",
         {-1.2, 1.},
         [](const std::vector<double>& x) {
             return 100 * std::pow((x[1] - x[0] * x[0]), 2) + std::pow((1 - x[0]), 2);
         }},
        {"Himmelblau's function",
         {-1., -1.},
         [](const std::vector<double>& x) {
             return std::pow(x[0] * x[0] + x[1] - 11, 2) + std::pow(x[0] + x[1] * x[1] - 7, 2);
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

    std::cout.precision(16);

    for (const auto& test : tests) {
        std::unique_ptr<optim::Method> opt(new optim::NelderMead);
        std::cout << test.name << std::endl;

        auto x = test.x0;
        opt->minimize(test.f, x, 1.e-8, 5000, std::cout);
    }

    return 0;
}
